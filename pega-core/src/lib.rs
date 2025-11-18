use std::{collections::HashMap, sync::Arc};

use cudarc::driver::CudaContext;
use tracing::{debug, info, instrument};

pub struct PegaEngine {
    context: Arc<CudaContext>,
    /// Store registered KV cache pointers (new IPC wrapper): layer_name -> registration
    kv_caches: HashMap<String, KVCacheRegistration>,
    /// Store saved KV blocks: (layer_name, block_hash) -> block data
    kv_storage: HashMap<(String, Vec<u8>), Block>,
}

#[derive(Debug, Clone)]
pub struct KVCacheRegistration {
    pub data_ptr: u64,
    pub size_bytes: usize,
    pub num_blocks: usize,
    pub bytes_per_block: usize,
    /// Distance in bytes between K and V segments when KV-first layout is used.
    /// Zero when the layout stores a single segment per block.
    pub kv_stride_bytes: usize,
    /// Number of segments per block (1 for blocks-first, 2 for KV-first).
    pub segments: usize,
}

#[derive(Clone)]
pub struct Block {
    pub data: Vec<u8>,
}

impl PegaEngine {
    /// Create a new PegaEngine instance
    #[instrument(level = "info")]
    pub fn new() -> Self {
        // default device is 0
        let context = cudarc::driver::CudaContext::new(0).unwrap();
        PegaEngine {
            context,
            kv_caches: HashMap::new(),
            kv_storage: HashMap::new(),
        }
    }

    /// Register a KV cache region with its layout info
    #[instrument(
        level = "info",
        skip(self),
        fields(layer = %layer_name, size_bytes, num_blocks, bytes_per_block)
    )]
    pub fn register_kv_cache(
        &mut self,
        layer_name: String,
        data_ptr: u64,
        size_bytes: usize,
        num_blocks: usize,
        bytes_per_block: usize,
        kv_stride_bytes: usize,
        segments: usize,
    ) {
        if bytes_per_block == 0 || num_blocks == 0 || segments == 0 {
            panic!("Invalid KV cache layout for layer {}", layer_name);
        }

        let registration = KVCacheRegistration {
            data_ptr,
            size_bytes,
            num_blocks,
            bytes_per_block,
            kv_stride_bytes,
            segments,
        };

        self.kv_caches.insert(layer_name, registration);
    }

    /// Unregister all KV cache handles
    #[instrument(level = "info", skip(self))]
    pub fn unregister_all_kv_caches(&mut self) {
        self.kv_caches.clear();
    }

    /// Get the number of registered KV caches
    #[instrument(level = "debug", skip(self), ret)]
    pub fn num_registered_kv_caches(&self) -> usize {
        self.kv_caches.len()
    }

    #[instrument(
        level = "debug",
        skip(self, block_ids, block_hashes),
        fields(layer = %layer_name, blocks = %block_ids.len(), hashes = %block_hashes.len()),
        err
    )]
    pub fn save_kv_blocks_from_ipc(
        &mut self,
        layer_name: String,
        block_ids: Vec<i32>,
        block_hashes: Vec<Vec<u8>>,
    ) -> Result<(), String> {
        if block_ids.len() != block_hashes.len() {
            return Err("block_ids and block_hashes must have equal length".into());
        }

        let Some(registration) = self.kv_caches.get(&layer_name) else {
            return Err(format!("Layer {} not registered", layer_name));
        };

        for (block_id, block_hash) in block_ids.into_iter().zip(block_hashes.into_iter()) {
            if block_id < 0 {
                continue;
            }
            let block_idx = block_id as usize;
            if block_idx >= registration.num_blocks {
                return Err(format!(
                    "Block {} out of range for layer {} ({} blocks registered)",
                    block_idx, layer_name, registration.num_blocks
                ));
            }

            // Copy each segment (K/V) for this block
            let mut combined = Vec::with_capacity(
                registration
                    .bytes_per_block
                    .checked_mul(registration.segments)
                    .ok_or_else(|| "Block size overflow".to_string())?,
            );
            for segment_idx in 0..registration.segments {
                let offset = self.segment_offset(&registration, block_idx, segment_idx)?;
                let mut buffer = vec![0u8; registration.bytes_per_block];
                self.copy_gpu_to_cpu(
                    registration.data_ptr,
                    offset,
                    &mut buffer,
                    registration.bytes_per_block,
                )?;
                combined.extend_from_slice(&buffer);
            }

            if self
                .kv_storage
                .contains_key(&(layer_name.clone(), block_hash.clone()))
            {
                continue;
            }

            info!("insert key {}-{:?} to kv_storage", layer_name, block_hash);

            self.kv_storage
                .insert((layer_name.clone(), block_hash), Block { data: combined });
        }

        Ok(())
    }

    /// Copy data from GPU to CPU
    #[instrument(level = "debug", skip(self, cpu_buffer), fields(offset, size), err)]
    fn copy_gpu_to_cpu(
        &self,
        gpu_base_ptr: u64,
        offset: usize,
        cpu_buffer: &mut [u8],
        size: usize,
    ) -> Result<(), String> {
        use cudarc::driver::sys;

        let src_ptr = gpu_base_ptr + offset as u64;
        let dst_ptr = cpu_buffer.as_mut_ptr();

        unsafe {
            // Use synchronous copy for simplicity
            let result = sys::cuMemcpyDtoH_v2(dst_ptr as *mut std::ffi::c_void, src_ptr, size);
            if result != sys::cudaError_enum::CUDA_SUCCESS {
                return Err(format!("cuMemcpyDtoH failed: {:?}", result));
            }
        }

        Ok(())
    }

    /// Get storage statistics
    /// Returns (num_blocks, total_bytes)
    #[instrument(level = "info", skip(self), ret)]
    pub fn get_storage_stats(&self) -> (usize, usize) {
        let num_blocks = self.kv_storage.len();
        let total_bytes: usize = self.kv_storage.values().map(|block| block.data.len()).sum();
        (num_blocks, total_bytes)
    }

    /// Check which KV blocks are available in CPU storage
    ///
    /// Args:
    ///   - layer_name: Name of the layer
    ///   - block_hashes: List of block hashes to check
    ///
    /// Returns:
    ///   - Vec<bool>: For each hash, true if available in storage
    #[instrument(
        level = "info",
        skip(self, block_hashes),
        fields(layer = %layer_name, requested = %block_hashes.len()),
        ret
    )]
    pub fn check_kv_blocks_availability(
        &self,
        layer_name: String,
        block_hashes: Vec<Vec<u8>>,
    ) -> Vec<bool> {
        let mut availability = Vec::with_capacity(block_hashes.len());

        for (idx, block_hash) in block_hashes.iter().enumerate() {
            let key = (layer_name.clone(), block_hash.clone());
            let available = self.kv_storage.contains_key(&key);
            availability.push(available);

            let hash_preview: Vec<u8> = block_hash.iter().copied().take(8).collect();
            debug!(
                block_index = idx,
                available,
                hash_prefix = ?hash_preview,
                "Checked KV block availability"
            );
        }

        let num_available = availability.iter().filter(|&&x| x).count();
        debug!(
            num_available,
            total = block_hashes.len(),
            "Completed KV block availability check"
        );

        availability
    }

    /// Load KV blocks from CPU memory to GPU via IPC handle
    ///
    /// Args:
    ///   - layer_name: Name of the layer
    ///   - block_ids: GPU block IDs to load into
    ///   - block_hashes: Content hashes for each block
    #[instrument(
        level = "info",
        skip(self, block_ids, block_hashes),
        fields(layer = %layer_name, blocks = %block_ids.len(), hashes = %block_hashes.len()),
        err
    )]
    pub fn load_kv_blocks_to_ipc(
        &self,
        layer_name: String,
        block_ids: Vec<i32>,
        block_hashes: Vec<Vec<u8>>,
    ) -> Result<(), String> {
        if block_ids.len() != block_hashes.len() {
            return Err("block_ids and block_hashes must have equal length".into());
        }

        let Some(registration) = self.kv_caches.get(&layer_name) else {
            return Err(format!("Layer {} not registered", layer_name));
        };

        for (block_id, block_hash) in block_ids.into_iter().zip(block_hashes.into_iter()) {
            if block_id < 0 {
                continue;
            }

            let block_idx = block_id as usize;
            if block_idx >= registration.num_blocks {
                return Err(format!(
                    "Block {} out of range for layer {} ({} blocks registered)",
                    block_idx, layer_name, registration.num_blocks
                ));
            }

            let key = (layer_name.clone(), block_hash.clone());
            info!("load key {}-{:?} from kv_storage", layer_name, block_hash);
            let Some(block) = self.kv_storage.get(&key) else {
                return Err(format!("Missing KV block for layer {}", layer_name));
            };

            let expected_size = registration
                .bytes_per_block
                .checked_mul(registration.segments)
                .ok_or_else(|| "Stored block size overflow".to_string())?;
            if block.data.len() != expected_size {
                return Err(format!(
                    "Stored block size mismatch for layer {}: {} vs {}",
                    layer_name,
                    block.data.len(),
                    expected_size
                ));
            }

            for segment_idx in 0..registration.segments {
                let offset = self.segment_offset(&registration, block_idx, segment_idx)?;
                let start = segment_idx * registration.bytes_per_block;
                let end = start + registration.bytes_per_block;
                let segment = &block.data[start..end];
                self.copy_cpu_to_gpu(
                    registration.data_ptr,
                    offset,
                    segment,
                    registration.bytes_per_block,
                )?;
            }
        }

        Ok(())
    }

    /// Calculate the byte offset for a given block/segment combination.
    fn segment_offset(
        &self,
        registration: &KVCacheRegistration,
        block_idx: usize,
        segment_idx: usize,
    ) -> Result<usize, String> {
        if segment_idx >= registration.segments {
            return Err("Segment index out of range".to_string());
        }

        let base = block_idx
            .checked_mul(registration.bytes_per_block)
            .ok_or_else(|| "Block offset overflow".to_string())?;

        let segment_offset = segment_idx
            .checked_mul(registration.kv_stride_bytes)
            .ok_or_else(|| "Segment offset overflow".to_string())?;

        let offset = base
            .checked_add(segment_offset)
            .ok_or_else(|| "Combined offset overflow".to_string())?;

        if offset + registration.bytes_per_block > registration.size_bytes {
            return Err(format!(
                "Block {} segment {} exceeds registered memory (offset {}, size {}, limit {})",
                block_idx,
                segment_idx,
                offset,
                registration.bytes_per_block,
                registration.size_bytes
            ));
        }

        Ok(offset)
    }

    /// Copy data from CPU to GPU
    #[instrument(level = "debug", skip(self, cpu_buffer), fields(offset, size), err)]
    fn copy_cpu_to_gpu(
        &self,
        gpu_base_ptr: u64,
        offset: usize,
        cpu_buffer: &[u8],
        size: usize,
    ) -> Result<(), String> {
        use cudarc::driver::sys;

        if cpu_buffer.len() < size {
            return Err(format!(
                "CPU buffer too small: {} bytes, need {} bytes",
                cpu_buffer.len(),
                size
            ));
        }

        let dst_ptr = gpu_base_ptr + offset as u64;
        let src_ptr = cpu_buffer.as_ptr();

        unsafe {
            // Use synchronous copy for simplicity
            let result = sys::cuMemcpyHtoD_v2(dst_ptr, src_ptr as *const std::ffi::c_void, size);
            if result != sys::cudaError_enum::CUDA_SUCCESS {
                return Err(format!("cuMemcpyHtoD failed: {:?}", result));
            }
        }

        Ok(())
    }
}

impl Default for PegaEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cudarc_basic() {
        // Get a stream for GPU 0
        let ctx = cudarc::driver::CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        // copy a rust slice to the device
        let _inp = stream.clone_htod(&[1.0f32; 100]).unwrap();

        // or allocate directly
        let _out = stream.alloc_zeros::<f32>(100).unwrap();
    }

    #[test]
    fn test_gpu_to_cpu_copy() {
        // 1. Create context and stream
        let ctx = cudarc::driver::CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        // 2. Allocate and initialize data on GPU
        let test_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let gpu_data = stream.clone_htod(&test_data).unwrap();

        // 3. Copy from GPU to CPU
        let cpu_data: Vec<f32> = stream.clone_dtoh(&gpu_data).unwrap();

        // 4. Verify the data
        assert_eq!(cpu_data, test_data);
        println!("GPU->CPU copy test passed! Data: {:?}", cpu_data);
    }

    #[test]
    fn test_gpu_to_cpu_copy_bf16() {
        // 1. Create context and stream
        let ctx = cudarc::driver::CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        // 2. Simulate KV cache block: [2, 16, 12, 64] * bf16 (2 bytes)
        let block_size = 2 * 16 * 12 * 64;
        let test_data: Vec<u8> = (0..block_size).map(|i| (i % 256) as u8).collect();

        // 3. Copy to GPU
        let gpu_block = stream.clone_htod(&test_data).unwrap();

        // 4. Copy back to CPU
        let cpu_block: Vec<u8> = stream.clone_dtoh(&gpu_block).unwrap();

        // 5. Verify
        assert_eq!(cpu_block.len(), block_size);
        assert_eq!(cpu_block, test_data);
        println!(
            "GPU->CPU BF16 block copy test passed! Block size: {} bytes",
            block_size
        );
    }
}
