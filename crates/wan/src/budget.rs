//! Physical memory admission, shared by every native pinned allocation.
use std::sync::{Mutex, atomic::{AtomicU64, AtomicUsize, Ordering}};
static FRACTION: AtomicU64 = AtomicU64::new(0);
pub static PINNED: AtomicUsize = AtomicUsize::new(0);
pub static ALLOCATION: Mutex<()> = Mutex::new(());
pub fn configure(fraction: f64) { FRACTION.store(fraction.to_bits(), Ordering::Relaxed); }
pub fn headroom(total: usize, available: usize, fraction: f64) -> usize {
    available.saturating_sub((total as f64 * fraction) as usize)
}
pub fn fits(available: usize, allocation: usize, reserve: usize) -> bool {
    allocation.checked_add(reserve).is_some_and(|n| n <= available)
}
pub fn available() -> crate::video::Result<usize> {
    let (total, free) = crate::measure::physical_ram()?;
    Ok(headroom(total, free, f64::from_bits(FRACTION.load(Ordering::Relaxed))))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reserve_is_of_total_and_existing_allocations_are_not_charged_twice() {
        assert_eq!(headroom(16 << 30, 3 << 30, 0.125), 1 << 30);
        assert_eq!(headroom(16 << 30, 1 << 30, 0.125), 0);
        assert_eq!(headroom(16 << 30, 4 << 30, 0.125), 2 << 30);
    }
    #[test]
    fn transition_respects_reserve_and_peak_not_final_size() {
        assert!(!fits(5, 4, 2));
        assert!(fits(6, 4, 2));
        assert!(!fits(usize::MAX, usize::MAX, 1));
    }
}
