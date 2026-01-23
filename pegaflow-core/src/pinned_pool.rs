use std::{
    num::NonZeroU64,
    ptr::NonNull,
    sync::{Arc, Mutex},
};

use bytesize::ByteSize;
use log::{error, info};

use crate::allocator::{Allocation, ScaledOffsetAllocator};
use crate::metrics::core_metrics;
use crate::pinned_mem::PinnedMemory;
use std::sync::atomic::{AtomicI64, Ordering};

/// RAII guard for a pinned memory allocation.
/// Automatically frees the allocation when dropped.
pub struct PinnedAllocation {
    allocation: Allocation,
    ptr: NonNull<u8>,
    pool: Arc<PinnedMemoryPool>,
}

// SAFETY: PinnedAllocation points to CUDA pinned memory which is fixed in physical
// memory and safe to access from any thread. The NonNull<u8> is just a pointer to
// this pinned memory region.
unsafe impl Send for PinnedAllocation {}
unsafe impl Sync for PinnedAllocation {}

impl PinnedAllocation {
    /// Get a const pointer to the allocated memory
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    /// Get a mutable pointer to the allocated memory
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Get the size of the allocation in bytes
    pub fn size(&self) -> NonZeroU64 {
        self.allocation.size_bytes
    }
}

impl Drop for PinnedAllocation {
    fn drop(&mut self) {
        // Automatically free the allocation when the guard is dropped
        self.pool.free_internal(&self.allocation);
    }
}

/// Manages a CUDA pinned memory pool and a byte-addressable allocator.
pub struct PinnedMemoryPool {
    /// Backing pinned memory (handles mmap + cudaHostRegister)
    backing: PinnedMemory,
    allocator: Mutex<ScaledOffsetAllocator>,
    // Tracks the last exported "largest free" value so we can drive an UpDownCounter like a gauge.
    last_largest_free_bytes_i64: AtomicI64,
}

impl PinnedMemoryPool {
    /// Upper bound for simultaneous allocations in the pinned pool.
    const MAX_ALLOCS: u32 = 4_000_000;
    /// Alignment for unit size (512 bytes for Direct I/O compatibility)
    const UNIT_ALIGNMENT: u64 = 512;

    /// Calculate unit size that fits pool_size into u32 range
    fn compute_unit_size(pool_size: u64, hint: Option<NonZeroU64>) -> u64 {
        let max_units = u32::MAX as u64;
        let min_unit_for_capacity = pool_size.div_ceil(max_units);
        let base = hint.map(|h| h.get()).unwrap_or(Self::UNIT_ALIGNMENT);
        let unit = base.max(min_unit_for_capacity);
        unit.div_ceil(Self::UNIT_ALIGNMENT) * Self::UNIT_ALIGNMENT
    }

    /// Allocate a new pinned memory pool of `pool_size` bytes.
    ///
    /// If `use_hugepages` is true, uses huge pages (requires system config).
    /// If `unit_size_hint` is provided, the allocator rounds allocations up to this size.
    pub fn new(pool_size: usize, use_hugepages: bool, unit_size_hint: Option<NonZeroU64>) -> Self {
        if pool_size == 0 {
            panic!("Pinned memory pool size must be greater than zero");
        }

        let backing = if use_hugepages {
            info!("Allocating pinned memory pool with huge pages");
            PinnedMemory::allocate_hugepages(pool_size)
                .expect("Failed to allocate pinned memory pool with huge pages")
        } else {
            info!("Allocating pinned memory pool with regular pages");
            PinnedMemory::allocate(pool_size).expect("Failed to allocate pinned memory pool")
        };

        let actual_size = backing.size() as u64;
        let unit_size = Self::compute_unit_size(actual_size, unit_size_hint);

        info!(
            "Pinned pool: size={}, unit_size={}, max_units={}",
            ByteSize(actual_size),
            ByteSize(unit_size),
            actual_size.div_ceil(unit_size)
        );

        let metrics = core_metrics();
        if let Ok(capacity_i64) = i64::try_from(actual_size) {
            metrics.pool_capacity_bytes.add(capacity_i64, &[]);
        } else {
            error!(
                "Pinned pool capacity exceeds i64::MAX; skipping capacity metric update: capacity_bytes={}",
                actual_size
            );
        }

        let allocator = ScaledOffsetAllocator::new_with_unit_size_and_max_allocs(
            actual_size,
            unit_size,
            Self::MAX_ALLOCS,
        )
        .unwrap_or_else(|err| {
            panic!(
                "Failed to create memory allocator (size={}, unit={}): {}",
                ByteSize(actual_size),
                ByteSize(unit_size),
                err
            )
        });

        let pool = Self {
            backing,
            allocator: Mutex::new(allocator),
            last_largest_free_bytes_i64: AtomicI64::new(0),
        };

        // Initialize fragmentation signal gauge (largest free == total at startup).
        pool.update_largest_free_metric(actual_size);

        pool
    }

    fn update_largest_free_metric(&self, largest_free_bytes: u64) {
        let Ok(new_i64) = i64::try_from(largest_free_bytes) else {
            // If the pool is too large to represent in i64, skip this metric.
            return;
        };

        let old_i64 = self
            .last_largest_free_bytes_i64
            .swap(new_i64, Ordering::Relaxed);
        let delta = new_i64 - old_i64;
        if delta != 0 {
            core_metrics().pool_largest_free_bytes.add(delta, &[]);
        }
    }

    /// Allocate pinned memory from the pool. Returns None when the allocation cannot be satisfied.
    /// Returns a RAII guard that automatically frees the allocation when dropped.
    pub fn allocate(self: &Arc<Self>, size: NonZeroU64) -> Option<PinnedAllocation> {
        let mut allocator = self.allocator.lock().unwrap();

        let allocation = match allocator.allocate(size.get()) {
            Ok(Some(allocation)) => allocation,
            Ok(None) => {
                return None; // Pool exhausted, caller can retry after eviction
            }
            Err(err) => {
                error!(
                    "Pinned memory allocation error: {} (requested {}): requested_bytes={}",
                    err,
                    ByteSize(size.get()),
                    size.get()
                );
                return None;
            }
        };

        // Update fragmentation signal after allocator mutation.
        self.update_largest_free_metric(allocator.storage_report().largest_free_allocation_bytes);

        let metrics = core_metrics();

        let offset: usize = allocation
            .offset_bytes
            .try_into()
            .expect("allocation offset exceeds usize");
        let ptr = unsafe { self.backing.as_ptr().add(offset) };
        let ptr = NonNull::new(ptr as *mut u8).expect("PinnedMemoryPool returned null pointer");

        let size_bytes = allocation.size_bytes.get();
        if let Ok(size_i64) = i64::try_from(size_bytes) {
            metrics.pool_used_bytes.add(size_i64, &[]);
        }

        Some(PinnedAllocation {
            allocation,
            ptr,
            pool: Arc::clone(self),
        })
    }

    /// Internal method to free a pinned memory allocation.
    /// This is called automatically by PinnedAllocation's Drop implementation.
    /// Users should not call this directly - use PinnedAllocation RAII instead.
    pub(crate) fn free_internal(&self, allocation: &Allocation) {
        let mut allocator = self.allocator.lock().unwrap();
        allocator.free(allocation);

        // Update fragmentation signal after allocator mutation.
        self.update_largest_free_metric(allocator.storage_report().largest_free_allocation_bytes);

        let metrics = core_metrics();
        let size_bytes = allocation.size_bytes.get();
        if let Ok(size_i64) = i64::try_from(size_bytes) {
            metrics.pool_used_bytes.add(-size_i64, &[]);
        }
    }

    /// Get (used_bytes, total_bytes) for the pool.
    pub fn usage(&self) -> (u64, u64) {
        let allocator = self.allocator.lock().unwrap();
        let report = allocator.storage_report();
        let total = allocator.total_bytes();
        let used = total - report.total_free_bytes;
        (used, total)
    }

    /// Largest contiguous free region currently available, in bytes.
    pub fn largest_free_allocation(&self) -> u64 {
        let allocator = self.allocator.lock().unwrap();
        allocator.storage_report().largest_free_allocation_bytes
    }
}

impl Drop for PinnedMemoryPool {
    fn drop(&mut self) {
        let metrics = core_metrics();
        let capacity_bytes = self.backing.size();
        if let Ok(capacity_i64) = i64::try_from(capacity_bytes) {
            metrics.pool_capacity_bytes.add(-capacity_i64, &[]);
        } else {
            error!(
                "Pinned pool capacity exceeds i64::MAX; skipping capacity metric cleanup: capacity_bytes={}",
                capacity_bytes
            );
        }

        // Best-effort cleanup for gauge-like metric.
        let last = self.last_largest_free_bytes_i64.load(Ordering::Relaxed);
        if last != 0 {
            metrics.pool_largest_free_bytes.add(-last, &[]);
        }
    }
}

// PinnedMemory handles cleanup in its Drop impl, no manual Drop needed here.

// SAFETY: The pool owns a PinnedMemory backing that remains valid for the lifetime
// of the pool. All mutations of the allocator state are guarded by the internal
// `Mutex`. CUDA pinned host memory can be accessed from any host thread.
unsafe impl Send for PinnedMemoryPool {}
unsafe impl Sync for PinnedMemoryPool {}
