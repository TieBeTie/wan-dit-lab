//! Timing model for the native producer, copy stream and compute stream.

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Candidate {
    pub host: usize,
    pub device: usize,
    pub host_bytes: usize,
    pub device_bytes: usize,
    pub seconds: f64,
}

/// Three cyclic passes: omit the first pass's startup. Costs are in milliseconds.
/// CPU cannot submit the next block until its synchronous refill has returned.
pub fn simulate(blocks: usize, host: usize, device: usize, read: f64, copy: f64, compute: f64, disk: bool) -> Option<f64> {
    if blocks == 0 || host == 0 || device == 0 || device > blocks
        || [read, copy, compute].iter().any(|x| !x.is_finite() || *x < 0.0) || compute == 0.0 { return None; }
    let depth = device;
    let mut cpu: f64 = 0.0;
    let mut copy_end: f64 = 0.0;
    let mut stage_free = vec![0.0f64; host];
    let mut ready = vec![0.0f64; 3 * blocks + depth];
    let mut finish = vec![0.0f64; 3 * blocks];
    let refill = |j: usize, free: f64, cpu: &mut f64, copy_end: &mut f64, stages: &mut Vec<f64>, ready: &mut Vec<f64>| {
        let stage = (j % blocks) % host;
        if disk { *cpu = cpu.max(stages[stage]) + read; }
        *copy_end = cpu.max(*copy_end).max(free) + copy;
        ready[j] = *copy_end;
        stages[stage] = *copy_end;
    };
    for j in 0..depth { refill(j, 0.0, &mut cpu, &mut copy_end, &mut stage_free, &mut ready); }
    for i in 0..3 * blocks {
        finish[i] = cpu.max(ready[i]).max(if i == 0 { 0.0 } else { finish[i - 1] }) + compute;
        refill(i + depth, finish[i], &mut cpu, &mut copy_end, &mut stage_free, &mut ready);
    }
    Some((finish[3 * blocks - 1] - finish[2 * blocks - 1]) / 1000.0)
}

pub fn choose(candidates: &[Candidate], host_budget: usize, device_budget: usize, frames: f64, fps_tolerance: f64) -> Option<Candidate> {
    if !frames.is_finite() || frames <= 0.0 || !fps_tolerance.is_finite() || fps_tolerance < 0.0 { return None; }
    let valid = |p: &&Candidate| p.seconds.is_finite() && p.seconds > 0.0 && p.host > 0 && p.device > 0
        && p.host_bytes <= host_budget && p.device_bytes <= device_budget;
    let best_fps = candidates.iter().filter(valid).map(|p| frames / p.seconds).fold(0.0f64, f64::max);
    candidates.iter().filter(valid).filter(|p| frames / p.seconds + 1e-10 * best_fps.max(1.0) >= best_fps - fps_tolerance)
        .min_by_key(|p| (p.host_bytes.saturating_add(p.device_bytes), p.device_bytes, p.host_bytes)).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn producer_stall_is_visible_even_when_dependency_wait_is_small() {
        let one = simulate(30, 1, 2, 8.0, 7.0, 10.0, true).unwrap();
        let two = simulate(30, 2, 2, 8.0, 7.0, 10.0, true).unwrap();
        assert!(one > two, "one={one}, two={two}");
    }

    #[test]
    fn extra_buffers_do_not_help_when_compute_already_hides_loading() {
        let one = simulate(30, 1, 2, 2.0, 7.0, 20.0, true).unwrap();
        let two = simulate(30, 2, 2, 2.0, 7.0, 20.0, true).unwrap();
        assert!((one - two).abs() < 1e-9);
    }

    #[test]
    fn selector_uses_fps_tolerance_and_both_memory_limits() {
        let slow = Candidate { host: 1, device: 2, host_bytes: 100, device_bytes: 200, seconds: 4.0 };
        let enough = Candidate { host: 2, device: 2, host_bytes: 200, device_bytes: 200, seconds: 2.0 };
        let extra = Candidate { host: 3, device: 3, host_bytes: 300, device_bytes: 300, seconds: 1.99 };
        assert_eq!(choose(&[slow, enough, extra], 1000, 1000, 32.0, 0.5), Some(enough));
        assert_eq!(choose(&[slow, enough, extra], 100, 1000, 32.0, 0.5), Some(slow));
        assert_eq!(choose(&[extra], 1000, 200, 32.0, 0.5), None);
    }

    #[test]
    fn rejects_invalid_costs_and_depths() {
        assert!(simulate(30, 0, 2, 1.0, 1.0, 1.0, true).is_none());
        assert!(simulate(30, 1, 2, f64::NAN, 1.0, 1.0, true).is_none());
    }
}
