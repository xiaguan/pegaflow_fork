//! Shared-memory synchronization state for async KV cache loading.
//!
//! Only the batch-level `LoadState` is supported. Other platforms are not
//! supported; compilation will fail outside x86_64 Linux.

#[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
compile_error!("LoadState is only supported on x86_64 Linux");

use shared_memory::{Shmem, ShmemConf, ShmemError};
use std::{
    alloc::Layout,
    ptr::NonNull,
    sync::atomic::{AtomicI64, AtomicU32, Ordering},
};
use uuid::Uuid;

/// State values for LoadState.
pub const LOAD_STATE_PENDING: i64 = 0;
pub const LOAD_STATE_SUCCESS: i64 = 1;
pub const LOAD_STATE_ERROR: i64 = -1;

/// Magic and version for LoadState header validation.
const LOAD_STATE_MAGIC: u32 = 0x5046_4c44; // 'PFLD'
pub const LOAD_STATE_VERSION: u32 = 1;

/// Bytes reserved at the start of the mapping to store the aligned offset.
const LOAD_STATE_META_SIZE: usize = std::mem::size_of::<usize>();
// Only support 64-bit systems for now.
const _: () = assert!(std::mem::size_of::<usize>() == 8);

#[repr(C)]
struct LoadStateHeader {
    magic: AtomicU32,
    version: AtomicU32,
}

/// In-memory layout for LoadState.
#[repr(C, align(8))]
struct LoadStateMem {
    header: LoadStateHeader,
    state: AtomicI64,
}

/// Batch-level synchronization state for async KV cache loading.
///
/// The connector creates this, passes the `shm_name` to the server, and
/// periodically polls `get()` to see whether the async load completed.
pub struct LoadState {
    shmem: Shmem,
    ptr: NonNull<LoadStateMem>,
}

// SAFETY: LoadState accesses shared memory exclusively through atomic fields.
// The pointer is validated for alignment and size on creation/attach.
unsafe impl Send for LoadState {}
unsafe impl Sync for LoadState {}

impl LoadState {
    /// Create a new LoadState (creates shared memory).
    ///
    /// The state is initialized to PENDING (0) and the header is stamped with
    /// a magic value and version for validation when attaching.
    pub fn new() -> Result<Self, ShmemError> {
        let shm_name = format!("pega_load_{}", Uuid::new_v4().as_simple());
        let shmem = ShmemConf::new()
            .os_id(&shm_name)
            .size(load_state_allocation_size())
            .create()?;

        let ptr = init_new_mapping(&shmem)?;
        let mem = unsafe { ptr.as_ref() };
        mem.header.magic.store(LOAD_STATE_MAGIC, Ordering::Release);
        mem.header
            .version
            .store(LOAD_STATE_VERSION, Ordering::Release);
        mem.state.store(LOAD_STATE_PENDING, Ordering::Release);

        Ok(Self { shmem, ptr })
    }

    /// Attach to an existing LoadState by shared memory name.
    pub fn attach(shm_name: &str) -> Result<Self, ShmemError> {
        let shmem = ShmemConf::new().os_id(shm_name).open()?;
        let ptr = attach_mapping(&shmem)?;
        let mem = unsafe { ptr.as_ref() };
        validate_header(mem)?;

        Ok(Self { shmem, ptr })
    }

    /// Get the shared memory identifier.
    pub fn shm_name(&self) -> &str {
        self.shmem.get_os_id()
    }

    /// Get current state value (non-blocking).
    pub fn get(&self) -> i64 {
        self.mem().state.load(Ordering::Acquire)
    }

    /// Set state to SUCCESS (1). Called by server when all transfers complete.
    pub fn set_completed(&self) {
        self.mem()
            .state
            .store(LOAD_STATE_SUCCESS, Ordering::Release);
    }

    /// Set state to ERROR (-1). Called by server on transfer failure.
    pub fn set_error(&self) {
        self.mem().state.store(LOAD_STATE_ERROR, Ordering::Release);
    }

    fn mem(&self) -> &LoadStateMem {
        // SAFETY: `ptr` is validated for size/alignment on creation/attach,
        // and the underlying memory lives for the lifetime of `shmem`.
        unsafe { self.ptr.as_ref() }
    }
}

fn load_state_layout() -> Layout {
    Layout::new::<LoadStateMem>()
}

fn load_state_allocation_size() -> usize {
    let layout = load_state_layout();
    LOAD_STATE_META_SIZE + layout.size() + layout.align()
}

fn align_up(value: usize, align: usize) -> Option<usize> {
    debug_assert!(align.is_power_of_two());
    let mask = align - 1;
    value.checked_add(mask).map(|v| v & !mask)
}

fn init_new_mapping(shmem: &Shmem) -> Result<NonNull<LoadStateMem>, ShmemError> {
    let layout = load_state_layout();
    if shmem.len() < LOAD_STATE_META_SIZE + layout.size() {
        return Err(ShmemError::MapSizeZero);
    }

    let base = shmem.as_ptr() as usize;
    let aligned = align_up(
        base.checked_add(LOAD_STATE_META_SIZE)
            .ok_or(ShmemError::MapSizeZero)?,
        layout.align(),
    )
    .ok_or(ShmemError::MapSizeZero)?;
    let offset = aligned.checked_sub(base).ok_or(ShmemError::MapSizeZero)?;

    if offset
        .checked_add(layout.size())
        .map_or(true, |end| end > shmem.len())
    {
        return Err(ShmemError::MapSizeZero);
    }

    // Record the offset so attach() can find the aligned region.
    unsafe {
        (shmem.as_ptr() as *mut usize).write_unaligned(offset);
    }

    if aligned % layout.align() != 0 {
        return Err(ShmemError::MapSizeZero);
    }

    NonNull::new(aligned as *mut LoadStateMem).ok_or(ShmemError::MapSizeZero)
}

fn attach_mapping(shmem: &Shmem) -> Result<NonNull<LoadStateMem>, ShmemError> {
    let layout = load_state_layout();
    if shmem.len() < LOAD_STATE_META_SIZE + layout.size() {
        return Err(ShmemError::MapSizeZero);
    }

    let base = shmem.as_ptr() as usize;
    let offset = unsafe { (shmem.as_ptr() as *const usize).read_unaligned() };

    if offset < LOAD_STATE_META_SIZE {
        return Err(ShmemError::MapSizeZero);
    }

    let ptr_addr = base.checked_add(offset).ok_or(ShmemError::MapSizeZero)?;

    if ptr_addr
        .checked_add(layout.size())
        .map_or(true, |end| end > shmem.len())
    {
        return Err(ShmemError::MapSizeZero);
    }

    if ptr_addr % layout.align() != 0 {
        return Err(ShmemError::MapSizeZero);
    }

    NonNull::new(ptr_addr as *mut LoadStateMem).ok_or(ShmemError::MapSizeZero)
}

fn validate_header(mem: &LoadStateMem) -> Result<(), ShmemError> {
    let magic = mem.header.magic.load(Ordering::Acquire);
    let version = mem.header.version.load(Ordering::Acquire);
    if magic != LOAD_STATE_MAGIC || version != LOAD_STATE_VERSION {
        return Err(ShmemError::MapSizeZero);
    }
    Ok(())
}
