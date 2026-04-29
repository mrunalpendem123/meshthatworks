//! Usage-adaptive expert caching for MeshThatWorks.
//!
//! Tracks which experts fire on which prompts (`ActivationHistogram`),
//! classifies each expert as hot / warm / cold (`ExpertTier`), and turns
//! those classifications into actual OS syscalls (`MemoryAdvisor`) so the
//! kernel keeps hot pages resident and lets cold pages drop. The bridge
//! between expert IDs and byte ranges is `parse_expert_layout` over the
//! model's safetensors file.
//!
//! The shape of the data is intentionally tiny — this crate decides
//! *policy*, not weights. The high-level flow:
//!
//! 1. `parse_expert_layout(model.safetensors)` → `ExpertId → ExpertLayout`
//! 2. `ActivationHistogram::record_prompt(...)` per generation step
//! 3. `apply_tiering(histogram, layout, advisor, thresholds)` issues
//!    `madvise(WILLNEED)` for hot experts and `madvise(DONTNEED)` for cold.
//!
//! This is the *outer* page-budget layer that complements SwiftLM's
//! MLX-internal `--stream-experts` eviction. Together they cap the
//! working set on small machines.

mod advisor;
mod safetensors_layout;

pub use advisor::{MemoryAdvisor, MemoryPolicy};
pub use safetensors_layout::{ExpertLayout, parse_expert_layout};

use std::collections::{HashMap, VecDeque};

/// (layer_index, expert_index_within_layer). Stable across a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ExpertId {
    pub layer: u32,
    pub expert: u32,
}

impl ExpertId {
    pub fn new(layer: u32, expert: u32) -> Self {
        Self { layer, expert }
    }
}

/// Caching tier for one expert. The engine maps these onto SwiftLM cache
/// hints: pinned → never evict, mmap → page on demand, stream → SSD load
/// on each access.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertTier {
    Hot,
    Warm,
    Cold,
}

/// Cutoffs for tiering — fraction of the rolling window in which the expert
/// was activated at least once. Tunable per model. Defaults are conservative.
#[derive(Debug, Clone, Copy)]
pub struct TierThresholds {
    pub hot_min: f32,
    pub warm_min: f32,
}

impl Default for TierThresholds {
    fn default() -> Self {
        // Hot: fired in ≥30% of recent prompts.
        // Warm: fired in ≥5%.
        // Otherwise cold.
        Self {
            hot_min: 0.30,
            warm_min: 0.05,
        }
    }
}

/// Rolling histogram: the per-expert hit count across the last `capacity`
/// prompts. Circular buffer; old prompts are decremented out as new ones
/// arrive. O(active_experts_per_prompt) per add/evict.
pub struct ActivationHistogram {
    capacity: usize,
    /// Per-prompt expert sets, oldest at front.
    window: VecDeque<Vec<ExpertId>>,
    /// Hit count = number of prompts in the window in which the expert fired.
    /// Multi-fire within a single prompt counts once.
    hits: HashMap<ExpertId, u32>,
}

impl ActivationHistogram {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "capacity must be > 0");
        Self {
            capacity,
            window: VecDeque::with_capacity(capacity),
            hits: HashMap::new(),
        }
    }

    /// Record one prompt's expert activations. Caller deduplicates within the
    /// prompt — the same expert firing 50 times in one prompt counts as one.
    pub fn record_prompt<I: IntoIterator<Item = ExpertId>>(&mut self, experts: I) {
        let mut deduped: Vec<ExpertId> = experts.into_iter().collect();
        deduped.sort();
        deduped.dedup();

        if self.window.len() == self.capacity {
            let evicted = self.window.pop_front().expect("non-empty by guard");
            for e in evicted {
                if let Some(c) = self.hits.get_mut(&e) {
                    *c = c.saturating_sub(1);
                    if *c == 0 {
                        self.hits.remove(&e);
                    }
                }
            }
        }

        for &e in &deduped {
            *self.hits.entry(e).or_insert(0) += 1;
        }
        self.window.push_back(deduped);
    }

    /// Number of prompts currently in the window. Always ≤ `capacity`.
    pub fn observed_prompts(&self) -> usize {
        self.window.len()
    }

    /// Hit rate ∈ [0.0, 1.0] for this expert across the rolling window.
    /// Returns 0 when the window is empty.
    pub fn hit_rate(&self, expert: ExpertId) -> f32 {
        if self.window.is_empty() {
            return 0.0;
        }
        let hits = self.hits.get(&expert).copied().unwrap_or(0);
        hits as f32 / self.window.len() as f32
    }

    /// Tier this expert under the given thresholds.
    pub fn tier(&self, expert: ExpertId, t: TierThresholds) -> ExpertTier {
        let r = self.hit_rate(expert);
        if r >= t.hot_min {
            ExpertTier::Hot
        } else if r >= t.warm_min {
            ExpertTier::Warm
        } else {
            ExpertTier::Cold
        }
    }

    /// Snapshot the current tier assignment for every expert that fired at
    /// least once in the rolling window. Cold experts that never fired are
    /// excluded — the engine treats "never observed" as cold by default.
    pub fn tiering(&self, t: TierThresholds) -> HashMap<ExpertId, ExpertTier> {
        self.hits
            .keys()
            .map(|&e| (e, self.tier(e, t)))
            .collect()
    }
}

/// Counts of experts moved into each tier by `apply_tiering`. Returned for
/// telemetry — useful in `[mtw-cache]` log lines.
#[derive(Debug, Clone, Default)]
pub struct TieringSummary {
    pub hot: u32,
    pub warm: u32,
    pub cold: u32,
    /// Bytes covered by HOT advisories, summed across all hot experts and
    /// their (possibly multi-range) layouts. Useful to confirm the hot
    /// working set fits the RAM budget.
    pub hot_bytes: u64,
    /// Bytes the cache asked the kernel to drop via `MADV_DONTNEED`.
    pub dropped_bytes: u64,
    /// Number of `advise` calls that errored — usually zero on macOS
    /// for valid ranges; counted for diagnostic visibility.
    pub advise_errors: u32,
}

/// Apply the histogram's current tier classification to the OS page cache
/// via `MemoryAdvisor`:
/// - Hot experts → `MADV_WILLNEED` on every byte range
/// - Warm experts → `MADV_NORMAL` (no opinion, let the kernel decide)
/// - Cold experts → `MADV_DONTNEED` (drop pages, free for hot)
///
/// Experts in `layout` but absent from the histogram are treated as cold
/// (kernel hasn't seen them yet — drop them so newly hot experts can be
/// resident).
///
/// Best-effort: a per-range advise failure is counted but does not abort
/// the whole pass. The cache classification is still useful even if the
/// kernel rejects some ranges.
pub fn apply_tiering(
    hist: &ActivationHistogram,
    layout: &HashMap<ExpertId, ExpertLayout>,
    advisor: &MemoryAdvisor,
    thresholds: TierThresholds,
) -> TieringSummary {
    let mut summary = TieringSummary::default();
    for (id, expert_layout) in layout {
        let tier = hist.tier(*id, thresholds);
        let policy = match tier {
            ExpertTier::Hot => MemoryPolicy::WillNeed,
            ExpertTier::Warm => MemoryPolicy::Normal,
            ExpertTier::Cold => MemoryPolicy::DontNeed,
        };
        match tier {
            ExpertTier::Hot => {
                summary.hot += 1;
                summary.hot_bytes += expert_layout.total_bytes();
            }
            ExpertTier::Warm => summary.warm += 1,
            ExpertTier::Cold => {
                summary.cold += 1;
                summary.dropped_bytes += expert_layout.total_bytes();
            }
        }
        for &(offset, len) in &expert_layout.ranges {
            if let Err(err) = advisor.advise(offset as usize, len as usize, policy) {
                tracing::debug!(?id, ?policy, %err, "apply_tiering: advise failed");
                summary.advise_errors += 1;
            }
        }
    }
    summary
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(l: u32, x: u32) -> ExpertId {
        ExpertId::new(l, x)
    }

    #[test]
    fn empty_histogram_returns_zero_hit_rate() {
        let h = ActivationHistogram::new(10);
        assert_eq!(h.hit_rate(e(0, 0)), 0.0);
        assert_eq!(h.tier(e(0, 0), TierThresholds::default()), ExpertTier::Cold);
    }

    #[test]
    fn dedup_within_prompt() {
        let mut h = ActivationHistogram::new(4);
        // expert (0,1) fires three times in one prompt → counts as 1.
        h.record_prompt([e(0, 1), e(0, 1), e(0, 1)]);
        assert_eq!(h.hit_rate(e(0, 1)), 1.0);
    }

    #[test]
    fn rolling_window_decrements() {
        let mut h = ActivationHistogram::new(2);
        h.record_prompt([e(0, 1)]);
        h.record_prompt([e(0, 1)]);
        assert_eq!(h.hit_rate(e(0, 1)), 1.0);
        h.record_prompt([e(0, 2)]);
        // (0,1) was in 1 of last 2 prompts.
        assert_eq!(h.hit_rate(e(0, 1)), 0.5);
        h.record_prompt([e(0, 2)]);
        // (0,1) fully aged out.
        assert_eq!(h.hit_rate(e(0, 1)), 0.0);
    }

    #[test]
    fn tiering_partitions_correctly() {
        let mut h = ActivationHistogram::new(10);
        // (0,0) fires every prompt (hot)
        // (0,1) fires every other (warm: 50% > 30% → also hot under default)
        // (0,2) fires once total (cold under default 5% threshold? 1/10 = 10% → warm)
        for i in 0..10 {
            let mut prompt = vec![e(0, 0)];
            if i % 2 == 0 {
                prompt.push(e(0, 1));
            }
            if i == 0 {
                prompt.push(e(0, 2));
            }
            h.record_prompt(prompt);
        }
        let t = TierThresholds::default();
        assert_eq!(h.tier(e(0, 0), t), ExpertTier::Hot);
        assert_eq!(h.tier(e(0, 1), t), ExpertTier::Hot); // 50% ≥ 30%
        assert_eq!(h.tier(e(0, 2), t), ExpertTier::Warm); // 10% ∈ [5%, 30%)
        assert_eq!(h.tier(e(0, 99), t), ExpertTier::Cold); // never seen
    }

    #[test]
    fn apply_tiering_classifies_and_summarizes() {
        // Build a histogram where (0,0) is hot, (0,1) is warm, (0,2) is cold.
        let mut h = ActivationHistogram::new(10);
        for i in 0..10 {
            let mut prompt = vec![e(0, 0)];
            if i % 2 == 0 {
                prompt.push(e(0, 1));
            }
            if i == 0 {
                prompt.push(e(0, 2));
            }
            h.record_prompt(prompt);
        }

        // Synth safetensors with three experts.
        use std::io::Write;
        let dir = std::env::temp_dir();
        let path = dir.join(format!("mtw-apply-{}.safetensors", std::process::id()));
        let header = serde_json::json!({
            "model.layers.0.mlp.experts.0.gate_proj.weight": {
                "dtype": "I8", "shape": [4, 4], "data_offsets": [0, 16]
            },
            "model.layers.0.mlp.experts.1.gate_proj.weight": {
                "dtype": "I8", "shape": [4, 4], "data_offsets": [16, 32]
            },
            "model.layers.0.mlp.experts.2.gate_proj.weight": {
                "dtype": "I8", "shape": [4, 4], "data_offsets": [32, 48]
            },
        })
        .to_string();
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        f.write_all(header.as_bytes()).unwrap();
        f.write_all(&vec![0u8; 48]).unwrap();
        f.sync_all().unwrap();

        let layout = parse_expert_layout(&path).unwrap();
        let advisor = MemoryAdvisor::open(&path).unwrap();
        let summary = apply_tiering(&h, &layout, &advisor, TierThresholds::default());

        assert_eq!(summary.hot, 2, "(0,0) and (0,1) are both ≥30%");
        assert_eq!(summary.warm, 1, "(0,2) at 10%");
        assert_eq!(summary.cold, 0);
        assert!(summary.hot_bytes > 0);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn tiering_snapshot_excludes_never_seen() {
        let mut h = ActivationHistogram::new(5);
        h.record_prompt([e(0, 0), e(1, 0)]);
        let snap = h.tiering(TierThresholds::default());
        assert_eq!(snap.len(), 2);
        assert!(snap.contains_key(&e(0, 0)));
        assert!(snap.contains_key(&e(1, 0)));
        assert!(!snap.contains_key(&e(0, 1)));
    }
}
