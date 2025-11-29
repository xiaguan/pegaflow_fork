//! Shared-memory synchronization state for async KV cache loading.
//!
//! This module provides two synchronization primitives:
//!
//! 1. `LayerSyncState` - Per-layer sync for layer-by-layer pipelining (legacy)
//! 2. `LoadState` - Batch-level sync for simpler synchronous waiting
//!
//! # Architecture (LoadState - recommended)
//!
//! ```text
//! Connector Worker                    Engine Server (async thread)
//! ─────────────────                   ────────────────────────────
//! create LoadState ──shm_name──▶      attach LoadState
//!
//! spin-wait on state ◄─────────────── set_completed() or set_error()
//! ```

use shared_memory::{Shmem, ShmemConf, ShmemError};
use std::sync::atomic::{AtomicI64, AtomicU8, Ordering};
use uuid::Uuid;

/// Flag values for layer synchronization state
const FLAG_NOT_STARTED: u8 = 0;
const FLAG_COMPLETED: u8 = 1;

/// State values for LoadState
pub const LOAD_STATE_PENDING: i64 = 0;
pub const LOAD_STATE_SUCCESS: i64 = 1;
pub const LOAD_STATE_ERROR: i64 = -1;

/// Batch-level synchronization state for async KV cache loading.
///
/// Much simpler than LayerSyncState - just a single atomic i64:
/// - 0: pending (load in progress)
/// - 1: success (all transfers complete)
/// - <0: error (transfer failed)
///
/// The connector creates this, passes the `shm_name` to the server,
/// and then spin-waits on the state until it becomes non-zero.
pub struct LoadState {
    shmem: Shmem,
}

// SAFETY: LoadState uses atomic operations for cross-process synchronization.
unsafe impl Send for LoadState {}
unsafe impl Sync for LoadState {}

impl LoadState {
    /// Create a new LoadState (creates shared memory).
    ///
    /// The state is initialized to PENDING (0).
    pub fn new() -> Result<Self, ShmemError> {
        let shm_name = format!("pega_load_{}", Uuid::new_v4().as_simple());
        let size = std::mem::size_of::<AtomicI64>();

        let shmem = ShmemConf::new().os_id(&shm_name).size(size).create()?;

        // Initialize to PENDING
        let ptr = shmem.as_ptr() as *mut AtomicI64;
        unsafe {
            (*ptr).store(LOAD_STATE_PENDING, Ordering::Relaxed);
        }

        Ok(Self { shmem })
    }

    /// Attach to an existing LoadState by shared memory name.
    pub fn attach(shm_name: &str) -> Result<Self, ShmemError> {
        let shmem = ShmemConf::new().os_id(shm_name).open()?;

        if shmem.len() < std::mem::size_of::<AtomicI64>() {
            return Err(ShmemError::MapSizeZero);
        }

        Ok(Self { shmem })
    }

    /// Get the shared memory identifier.
    pub fn shm_name(&self) -> &str {
        self.shmem.get_os_id()
    }

    /// Reset state to PENDING. Call before starting a new load batch.
    pub fn reset(&self) {
        let ptr = self.shmem.as_ptr() as *mut AtomicI64;
        unsafe {
            (*ptr).store(LOAD_STATE_PENDING, Ordering::Release);
        }
    }

    /// Get current state value (non-blocking).
    pub fn get(&self) -> i64 {
        let ptr = self.shmem.as_ptr() as *const AtomicI64;
        unsafe { (*ptr).load(Ordering::Acquire) }
    }

    /// Set state to SUCCESS (1). Called by server when all transfers complete.
    pub fn set_completed(&self) {
        let ptr = self.shmem.as_ptr() as *mut AtomicI64;
        unsafe {
            (*ptr).store(LOAD_STATE_SUCCESS, Ordering::Release);
        }
    }

    /// Set state to ERROR (-1). Called by server on transfer failure.
    pub fn set_error(&self) {
        let ptr = self.shmem.as_ptr() as *mut AtomicI64;
        unsafe {
            (*ptr).store(LOAD_STATE_ERROR, Ordering::Release);
        }
    }

    /// Spin-wait until state becomes non-zero (completed or error).
    ///
    /// Returns the final state value (1 for success, <0 for error).
    pub fn wait(&self) -> i64 {
        let ptr = self.shmem.as_ptr() as *const AtomicI64;
        loop {
            let state = unsafe { (*ptr).load(Ordering::Acquire) };
            if state != LOAD_STATE_PENDING {
                return state;
            }
            std::hint::spin_loop();
        }
    }
}

/// Shared-memory based synchronization state for async layer loading.
///
/// The connector creates this, passes the `shm_name` to the server,
/// and the server attaches to the same shared memory region.
///
/// NOTE: This is the legacy per-layer sync mechanism. Consider using
/// LoadState for simpler batch-level synchronization.
pub struct LayerSyncState {
    shmem: Shmem,
    num_layers: usize,
}

// SAFETY: LayerSyncState uses atomic operations for cross-process synchronization.
// The shared memory region is accessed via atomic operations only.
unsafe impl Send for LayerSyncState {}
unsafe impl Sync for LayerSyncState {}

impl LayerSyncState {
    /// Create a new LayerSyncState with the given number of layers.
    ///
    /// This creates a new shared memory region with a unique name.
    /// The name can be retrieved via `shm_name()` and passed to the server.
    pub fn new(num_layers: usize) -> Result<Self, ShmemError> {
        let shm_name = format!("pega_sync_{}", Uuid::new_v4().as_simple());
        let size = num_layers;

        let shmem = ShmemConf::new().os_id(&shm_name).size(size).create()?;

        // Initialize all flags to NOT_STARTED
        let ptr = shmem.as_ptr() as *mut AtomicU8;
        for i in 0..num_layers {
            unsafe {
                (*ptr.add(i)).store(FLAG_NOT_STARTED, Ordering::Relaxed);
            }
        }

        Ok(Self { shmem, num_layers })
    }

    /// Attach to an existing shared memory region by name.
    ///
    /// Used by the server to attach to the region created by the connector.
    pub fn attach(shm_name: &str, num_layers: usize) -> Result<Self, ShmemError> {
        let shmem = ShmemConf::new().os_id(shm_name).open()?;

        if shmem.len() < num_layers {
            return Err(ShmemError::MapSizeZero);
        }

        Ok(Self { shmem, num_layers })
    }

    /// Get the shared memory identifier.
    ///
    /// Pass this to the server so it can attach to the same region.
    pub fn shm_name(&self) -> &str {
        self.shmem.get_os_id()
    }

    /// Get the number of layers.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Reset all flags to NOT_STARTED.
    ///
    /// Call this before starting a new load batch.
    pub fn reset(&self) {
        let ptr = self.shmem.as_ptr() as *mut AtomicU8;
        for i in 0..self.num_layers {
            unsafe {
                (*ptr.add(i)).store(FLAG_NOT_STARTED, Ordering::Release);
            }
        }
    }

    /// Mark a layer as completed.
    ///
    /// Called by the server's async thread after the CUDA transfer is done.
    pub fn mark_layer_done(&self, layer_id: usize) {
        if layer_id >= self.num_layers {
            return;
        }
        let ptr = self.shmem.as_ptr() as *mut AtomicU8;
        unsafe {
            (*ptr.add(layer_id)).store(FLAG_COMPLETED, Ordering::Release);
        }
    }

    /// Wait until a layer is completed.
    ///
    /// Called by the connector worker before executing attention for that layer.
    /// Uses spin-wait for simplicity.
    pub fn wait_layer(&self, layer_id: usize) {
        if layer_id >= self.num_layers {
            return;
        }
        let ptr = self.shmem.as_ptr() as *const AtomicU8;
        loop {
            let flag = unsafe { (*ptr.add(layer_id)).load(Ordering::Acquire) };
            if flag == FLAG_COMPLETED {
                return;
            }
            std::hint::spin_loop();
        }
    }

    /// Check if a layer is completed (non-blocking).
    pub fn is_layer_done(&self, layer_id: usize) -> bool {
        if layer_id >= self.num_layers {
            return true;
        }
        let ptr = self.shmem.as_ptr() as *const AtomicU8;
        let flag = unsafe { (*ptr.add(layer_id)).load(Ordering::Acquire) };
        flag == FLAG_COMPLETED
    }
}
