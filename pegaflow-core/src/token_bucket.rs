// ============================================================================
// TokenBucket: Lock-free rate limiter inspired by Seastar's shared_token_bucket.
//
// Design: Two rovers (tail, head) chase each other on a wrapping counter.
// - grab(): advances tail, returns deficiency (how much over budget)
// - replenish(): advances head based on elapsed time
// - deficiency > 0 means caller should wait before proceeding
//
// Reference: seastar/include/seastar/core/shared_token_bucket.hh
// ============================================================================

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// A lock-free token bucket for rate limiting.
///
/// Tokens represent bytes (or any resource unit). The bucket refills at
/// `rate` tokens per second, up to `limit` capacity (burst size).
pub struct TokenBucket {
    /// Consumer position (advances on grab)
    tail: AtomicU64,
    /// Replenish position (advances on replenish)
    head: AtomicU64,
    /// Tokens per second
    rate: u64,
    /// Maximum burst capacity
    limit: u64,
    /// Minimum tokens to accumulate before replenishing (reduces CAS contention)
    threshold: u64,
    /// Last replenish timestamp in nanoseconds (from reference instant)
    last_replenish_nanos: AtomicU64,
    /// Reference instant for time calculations
    epoch: Instant,
}

impl TokenBucket {
    /// Create a new token bucket.
    ///
    /// # Arguments
    /// * `rate` - Tokens per second (e.g., 20 * 1024^3 for 20 GB/s)
    /// * `limit` - Burst capacity (max tokens that can accumulate)
    pub fn new(rate: u64, limit: u64) -> Self {
        let threshold = (limit / 10).max(1);
        Self {
            tail: AtomicU64::new(0),
            head: AtomicU64::new(limit), // Start with full bucket
            rate,
            limit,
            threshold,
            last_replenish_nanos: AtomicU64::new(0),
            epoch: Instant::now(),
        }
    }

    /// Create with custom threshold.
    pub fn with_threshold(rate: u64, limit: u64, threshold: u64) -> Self {
        Self {
            tail: AtomicU64::new(0),
            head: AtomicU64::new(limit),
            rate,
            limit,
            threshold: threshold.clamp(1, limit),
            last_replenish_nanos: AtomicU64::new(0),
            epoch: Instant::now(),
        }
    }

    /// Attempt to grab tokens. Returns deficiency (tokens over budget).
    ///
    /// - Returns 0: tokens available, proceed immediately
    /// - Returns N > 0: need to wait for N tokens worth of time
    ///
    /// This is lock-free and forms a natural queue: earlier callers
    /// get smaller deficiency values and wake up first.
    pub fn grab(&self, tokens: u64) -> u64 {
        let new_tail = self.tail.fetch_add(tokens, Ordering::Relaxed) + tokens;
        let head = self.head.load(Ordering::Relaxed);
        new_tail.saturating_sub(head)
    }

    /// Replenish tokens based on elapsed time. Call this periodically.
    ///
    /// Thread-safe: only one caller will succeed per time window.
    pub fn replenish(&self, now: Instant) {
        let now_nanos = now.duration_since(self.epoch).as_nanos() as u64;
        let last = self.last_replenish_nanos.load(Ordering::Relaxed);

        if now_nanos <= last {
            return;
        }

        let delta_nanos = now_nanos - last;
        // tokens = rate * delta_secs = rate * delta_nanos / 1e9
        let extra = self.tokens_for_duration_nanos(delta_nanos);

        if extra < self.threshold {
            return;
        }

        // Try to claim this replenish window
        if self
            .last_replenish_nanos
            .compare_exchange_weak(last, now_nanos, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
        {
            return; // Another thread won, try next time
        }

        // Don't let head exceed tail + limit (cap the burst)
        let max_extra = self.max_replenish();
        self.head.fetch_add(extra.min(max_extra), Ordering::Relaxed);
    }

    /// Convenience: grab with auto-replenish.
    pub fn grab_with_replenish(&self, tokens: u64) -> u64 {
        self.replenish(Instant::now());
        self.grab(tokens)
    }

    /// Calculate how long to wait for a given deficiency.
    pub fn duration_for(&self, deficiency: u64) -> Duration {
        if deficiency == 0 || self.rate == 0 {
            return Duration::ZERO;
        }
        // duration = deficiency / rate (in seconds)
        // = deficiency * 1e9 / rate (in nanos)
        let nanos = (deficiency as u128 * 1_000_000_000) / self.rate as u128;
        Duration::from_nanos(nanos as u64)
    }

    /// Current available tokens (may be negative conceptually, returns 0 if so).
    pub fn available(&self) -> u64 {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);
        head.saturating_sub(tail)
    }

    /// Refund tokens (e.g., if operation was cancelled).
    pub fn refund(&self, tokens: u64) {
        self.head.fetch_add(tokens, Ordering::Relaxed);
    }

    // Internal: max tokens that can be added without exceeding limit
    fn max_replenish(&self) -> u64 {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Relaxed);
        (tail + self.limit).saturating_sub(head)
    }

    // Internal: calculate tokens for a duration in nanoseconds
    fn tokens_for_duration_nanos(&self, nanos: u64) -> u64 {
        ((self.rate as u128 * nanos as u128) / 1_000_000_000) as u64
    }

    /// Get the configured rate (tokens/sec).
    pub fn rate(&self) -> u64 {
        self.rate
    }

    /// Get the configured limit (burst capacity).
    pub fn limit(&self) -> u64 {
        self.limit
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_basic_grab_and_availability() {
        // 1000 tokens/sec, 100 token burst
        let bucket = TokenBucket::new(1000, 100);

        // Initially full
        assert_eq!(bucket.available(), 100);

        // Grab 30 tokens, should succeed (deficiency = 0)
        let def = bucket.grab(30);
        assert_eq!(def, 0);
        assert_eq!(bucket.available(), 70);

        // Grab 50 more
        let def = bucket.grab(50);
        assert_eq!(def, 0);
        assert_eq!(bucket.available(), 20);

        // Grab 30 more - exceeds available, deficiency = 10
        let def = bucket.grab(30);
        assert_eq!(def, 10);
        assert_eq!(bucket.available(), 0);
    }

    #[test]
    fn test_replenish_over_time() {
        // 1000 tokens/sec, 100 token burst, threshold=1 for testing
        let bucket = TokenBucket::with_threshold(1000, 100, 1);

        // Drain the bucket
        bucket.grab(100);
        assert_eq!(bucket.available(), 0);

        // Sleep 50ms -> should replenish ~50 tokens
        thread::sleep(Duration::from_millis(50));
        bucket.replenish(Instant::now());

        let avail = bucket.available();
        // Allow some timing slack: expect 40-60 tokens
        assert!(
            (40..=60).contains(&avail),
            "expected 40-60 tokens after 50ms, got {}",
            avail
        );
    }

    #[test]
    fn test_duration_for_deficiency() {
        // 1_000_000 tokens/sec (1 MB/s)
        let bucket = TokenBucket::new(1_000_000, 1000);

        // 1000 tokens deficiency at 1MB/s = 1ms
        let dur = bucket.duration_for(1000);
        assert_eq!(dur, Duration::from_millis(1));

        // 500_000 tokens = 500ms
        let dur = bucket.duration_for(500_000);
        assert_eq!(dur, Duration::from_millis(500));

        // 0 deficiency = no wait
        let dur = bucket.duration_for(0);
        assert_eq!(dur, Duration::ZERO);
    }

    #[test]
    fn test_burst_limit_respected() {
        let bucket = TokenBucket::with_threshold(1000, 100, 1);

        // Drain and wait long enough to "over-replenish"
        bucket.grab(100);
        thread::sleep(Duration::from_millis(200)); // Would be 200 tokens at 1000/s
        bucket.replenish(Instant::now());

        // Should cap at limit (100), not 200
        assert!(
            bucket.available() <= 100,
            "bucket should not exceed limit, got {}",
            bucket.available()
        );
    }

    #[test]
    fn test_refund() {
        let bucket = TokenBucket::new(1000, 100);

        bucket.grab(50);
        assert_eq!(bucket.available(), 50);

        // Refund 20
        bucket.refund(20);
        assert_eq!(bucket.available(), 70);
    }
}
