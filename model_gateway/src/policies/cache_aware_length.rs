/*
    Cache-Aware Length Load Balancing Router (cache_aware_length)

    Splits healthy workers into a "long pool" and a "short pool" by the
    `pool` worker label (`pool=long` → long pool, otherwise short pool), then
    applies cache-affinity routing on top of the split. Designed for P/D
    disaggregated prefill fleets and regular single-node fleets alike — pool
    membership is label-driven and independent of `WorkerType`.

    Routing pipeline (5 steps), mirroring `cache_aware` for steps 1-3 and
    adding a long/short split at step 4:

    Step 1 - Health filter
        is_available()==false workers are dropped; empty → 503 (None), no tree.
    Step 2 - Global imbalance check (same formula as cache_aware)
        (max_load - min_load) > abs_threshold  AND
        max_load > min_load * rel_threshold
          YES → route to healthy min-load, record tree, return.
          NO  → continue.
    Step 3 - Cache hit check (approximate string tree, char-level)
        tree missing → random healthy worker, no tree (init race).
        match_rate = matched_chars / input_chars
        match_rate > cache_threshold
          YES → hit branch: route to the highest-matching worker regardless of
                pool; if that worker is unhealthy, clean its stale tenant and
                fall back to the first healthy worker. Record tree.
          NO  → continue to step 4.
    Step 4 - No-cache branch: split by uncached prefill tokens
        token source (priority):
          1. X-Prompt-Tokens header (exact, supplied by an upstream gateway).
          2. (input_chars - matched_chars) / chars_per_token (char estimate).
          3. neither computable → all-healthy min-load, record tree.
        long pool  = healthy workers with labels["pool"] == "long"
        short pool = remaining healthy workers
        uncached >= long_prefill_threshold (long request):
            long pool has free worker (load < long_pool_max_load)
                → long pool min-load
            else short pool has an idle worker (load == 0)
                → that worker (long→short overflow)
            else long pool has a healthy worker
                → long pool min-load (queue)
            else → all-healthy min-load
        uncached < long_prefill_threshold (short request):
            short pool has free worker (load < short_pool_max_load)
                → short pool min-load
            else long pool has free worker
                → long pool min-load (short→long overflow)
            else short pool has a worker
                → short pool min-load (fallback queue)
            else long pool has a worker
                → long pool min-load
            else (both pools empty) → all-healthy min-load
    Step 5 - Record tree + return
        tree.insert_text(text, selected worker url) and increment_processed()
        for every selection except the 503 and the no-tree random fallback.

    Configuration Parameters:
    ------------------------
    cache_threshold:        Min prefix match ratio for hit routing (0.0-1.0)
    balance_abs_threshold: Absolute load diff for global imbalance detection
    balance_rel_threshold: Relative load ratio for global imbalance detection
    eviction_interval_secs: Interval between LRU eviction cycles
    max_tree_size:         Max total chars of each model's approximate tree
    chars_per_token:       Divisor for char-level token estimation (default 4)
    long_prefill_threshold: Uncached-token boundary between long and short
    long_pool_max_load:    Load ceiling for the long pool
    short_pool_max_load:   Load ceiling for the short pool
*/

use std::sync::Arc;

use dashmap::DashMap;
use kv_index::{PrefixMatchResult, TenantId, Tree};
use rand::RngExt;
use tracing::debug;

use super::{
    normalize_model_key, utils::PeriodicTask, CacheAwareLengthConfig, LoadBalancingPolicy,
    SelectWorkerInfo,
};
use crate::{observability::metrics::Metrics, worker::Worker};

/// HTTP header carrying the exact prompt token count, supplied by an
/// upstream gateway that has already tokenized the request. Case-insensitive
/// per the `http` crate's `HeaderName` equality.
const HEADER_PROMPT_TOKENS: &str = "x-prompt-tokens";

/// Cache-aware length routing policy.
///
/// Routes requests based on cache affinity and a long/short pool split. The
/// split is driven by the `pool` worker label (`pool=long`); workers without
/// the label form the short pool. Self-contained: maintains only a
/// per-model string radix tree and a background eviction task. Does not
/// participate in mesh tree sync, KV-event monitoring, or the hash index —
/// those are handled by `CacheAwarePolicy`.
#[derive(Debug)]
pub struct CacheAwareLengthPolicy {
    config: CacheAwareLengthConfig,
    /// String-based trees for HTTP connections (text input), keyed by model.
    string_trees: Arc<DashMap<String, Arc<Tree>>>,
    _eviction_task: Option<PeriodicTask>,
}

impl Default for CacheAwareLengthPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl CacheAwareLengthPolicy {
    pub fn new() -> Self {
        Self::with_config(CacheAwareLengthConfig::default())
    }

    pub fn with_config(config: CacheAwareLengthConfig) -> Self {
        let string_trees = Arc::new(DashMap::<String, Arc<Tree>>::new());

        let eviction_task = (config.eviction_interval_secs > 0).then(|| {
            let trees_clone = Arc::clone(&string_trees);
            let max_tree_size = config.max_tree_size;
            PeriodicTask::spawn(
                config.eviction_interval_secs,
                "LengthTreeEviction",
                move || {
                    let mut total_chars = 0usize;
                    for tree_ref in trees_clone.iter() {
                        let model_id = tree_ref.key();
                        let tree = tree_ref.value();
                        tree.evict_tenant_by_size(max_tree_size);

                        let counts = tree.get_tenant_char_count();
                        let chars: usize = counts.values().sum();
                        let tenants = counts.len();
                        total_chars += chars;
                        Metrics::set_cache_tree_chars(model_id, chars);
                        Metrics::set_cache_tree_tenants(model_id, "string", tenants);

                        debug!(
                            "String tree eviction completed for model {}, max_size: {}",
                            model_id, max_tree_size
                        );
                    }
                    tracing::info!(
                        "Length tree memory: string_trees={} models / {} chars",
                        trees_clone.len(),
                        total_chars,
                    );
                },
            )
        });

        Self {
            config,
            string_trees,
            _eviction_task: eviction_task,
        }
    }

    /// Initialize trees for a set of workers (seed each worker as a tenant of
    /// the root so it is always a cache-hit candidate). Call after workers are
    /// registered.
    pub fn init_workers(&self, workers: &[Arc<dyn Worker>]) {
        // Group workers by model
        let mut model_workers: std::collections::HashMap<String, Vec<&Arc<dyn Worker>>> =
            std::collections::HashMap::new();
        for worker in workers {
            let tree_key = normalize_model_key(worker.model_id());
            model_workers
                .entry(tree_key.to_string())
                .or_default()
                .push(worker);
        }

        for (tree_key, model_workers) in model_workers {
            let string_tree = self
                .string_trees
                .entry(tree_key)
                .or_insert_with(|| Arc::new(Tree::new()));
            for worker in model_workers {
                string_tree.insert_text("", worker.url());
            }
        }
    }

    /// Add a single worker to the trees (incremental update).
    pub fn add_worker(&self, worker: &dyn Worker) {
        let tree_key = normalize_model_key(worker.model_id()).to_string();
        let string_tree = self
            .string_trees
            .entry(tree_key)
            .or_insert_with(|| Arc::new(Tree::new()));
        string_tree.insert_text("", worker.url());
    }

    /// Remove a worker from the trees, purging its tenant from every model's
    /// string tree. A removed worker's tenant count never grows again, so
    /// size-based eviction alone would retain its subtree forever.
    pub fn remove_worker_by_url(&self, url: &str) {
        let tenant: TenantId = Arc::from(url);
        for tree_ref in self.string_trees.iter() {
            tree_ref.value().remove_tenant_all(&tenant);
        }
    }
}

impl LoadBalancingPolicy for CacheAwareLengthPolicy {
    fn select_worker(&self, workers: &[Arc<dyn Worker>], info: &SelectWorkerInfo) -> Option<usize> {
        let text = info.request_text.unwrap_or("");

        // Step 1: health filter — single O(workers) gather reading each worker
        // once via routing_state() (health + load + processed under one guard).
        let mut healthy_indices: Vec<usize> = Vec::with_capacity(workers.len());
        let mut min_key: Option<(usize, usize, usize)> = None;
        let mut min_load_idx: Option<usize> = None;
        for (idx, worker) in workers.iter().enumerate() {
            let state = worker.routing_state();
            if state.eligible() {
                healthy_indices.push(idx);
                let key = (state.load, state.processed, idx);
                match min_key {
                    Some(best) if key >= best => {}
                    _ => {
                        min_key = Some(key);
                        min_load_idx = Some(idx);
                    }
                }
            }
        }

        if healthy_indices.is_empty() {
            return None; // 503, do not record tree
        }

        let model_id = normalize_model_key(workers[healthy_indices[0]].model_id()).to_string();

        // Step 2: global imbalance check (same formula as cache_aware).
        // The min/max are over the healthy fleet only.
        let healthy_min = min_key.map(|(load, _, _)| load).unwrap_or(0);
        let healthy_max = healthy_indices
            .iter()
            .map(|&i| workers[i].routing_state().load)
            .max()
            .unwrap_or(0);
        let abs_diff = healthy_max.saturating_sub(healthy_min);
        let rel_threshold = self.config.balance_rel_threshold * healthy_min as f32;
        if abs_diff > self.config.balance_abs_threshold && healthy_max as f32 > rel_threshold {
            // min_load_idx is guaranteed Some here (healthy_indices non-empty
            // populates it in the same loop), but `?` keeps the analyzer happy
            // without a deny-listed unwrap.
            let selected = min_load_idx?;
            self.record_tree(&model_id, text, workers[selected].url());
            workers[selected].increment_processed();
            debug!(
                branch = "global_imbalance_min_load",
                worker = workers[selected].url(),
                model_id = model_id,
                "cache_aware_length selection"
            );
            return Some(selected);
        }

        // Step 3: cache hit check (approximate string tree, char-level).
        let tree = self
            .string_trees
            .get(&model_id)
            .map(|entry| entry.value().clone());

        let Some(tree) = tree else {
            // tree missing: init race — random healthy worker, do not record.
            let idx = healthy_indices[rand::rng().random_range(0..healthy_indices.len())];
            debug!(
                branch = "no_tree_random",
                worker = workers[idx].url(),
                model_id = model_id,
                "cache_aware_length selection"
            );
            return Some(idx);
        };

        let result = tree.match_prefix_with_counts(text);
        let match_rate = if result.input_char_count == 0 {
            0.0
        } else {
            result.matched_char_count as f32 / result.input_char_count as f32
        };

        if match_rate > self.config.cache_threshold {
            // Cache hit: route to the highest-matching worker regardless of pool.
            if let Some(idx) = Self::select_matched_candidate(workers, &healthy_indices, &result) {
                self.record_tree(&model_id, text, workers[idx].url());
                workers[idx].increment_processed();
                debug!(
                    branch = "cache_hit",
                    worker = workers[idx].url(),
                    match_rate,
                    model_id = model_id,
                    "cache_aware_length selection"
                );
                return Some(idx);
            }
            // Hit but the matched worker is unhealthy: clean stale tenant and
            // fall back to the first healthy worker (record tree).
            if let Some(tenant) = result.matched_tenants.first() {
                tree.remove_tenant_all(tenant);
            }
            let idx = healthy_indices[0];
            self.record_tree(&model_id, text, workers[idx].url());
            workers[idx].increment_processed();
            debug!(
                branch = "hit_unhealthy_first_healthy",
                worker = workers[idx].url(),
                model_id = model_id,
                "cache_aware_length selection"
            );
            return Some(idx);
        }

        // Step 4: no-cache branch — split by uncached prefill tokens.
        let uncached_tokens = self.compute_uncached_tokens(info, &result);
        let Some(uncached) = uncached_tokens else {
            // Neither source computable → all-healthy min-load.
            let selected = min_load_idx?;
            self.record_tree(&model_id, text, workers[selected].url());
            workers[selected].increment_processed();
            debug!(
                branch = "uncached_unknown_min_load",
                worker = workers[selected].url(),
                model_id = model_id,
                "cache_aware_length selection"
            );
            return Some(selected);
        };

        let long_indices: Vec<usize> = healthy_indices
            .iter()
            .copied()
            .filter(|&i| is_long_pool(&*workers[i]))
            .collect();
        let short_indices: Vec<usize> = healthy_indices
            .iter()
            .copied()
            .filter(|&i| !is_long_pool(&*workers[i]))
            .collect();

        let selected = if uncached >= self.config.long_prefill_threshold {
            self.select_long_request(
                workers,
                &long_indices,
                &short_indices,
                &healthy_indices,
                min_load_idx,
            )
        } else {
            self.select_short_request(
                workers,
                &long_indices,
                &short_indices,
                &healthy_indices,
                min_load_idx,
            )
        };

        // Step 5: record tree + return.
        let selected = selected.unwrap_or_else(|| min_load_idx.unwrap_or(healthy_indices[0]));
        self.record_tree(&model_id, text, workers[selected].url());
        workers[selected].increment_processed();
        debug!(
            branch = "pool_split",
            worker = workers[selected].url(),
            uncached_tokens = uncached,
            is_long = uncached >= self.config.long_prefill_threshold,
            model_id = model_id,
            "cache_aware_length selection"
        );
        Some(selected)
    }

    fn name(&self) -> &'static str {
        "cache_aware_length"
    }

    fn needs_request_text(&self) -> bool {
        true
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// Private helper methods
impl CacheAwareLengthPolicy {
    /// Record a routing decision into the model's string tree.
    fn record_tree(&self, model_id: &str, text: &str, worker_url: &str) {
        if let Some(tree) = self.string_trees.get(model_id).map(|e| e.value().clone()) {
            tree.insert_text(text, worker_url);
        }
    }

    /// Pressure-select among the tenants holding the matched prefix. The
    /// match is on the raw prefix, so every matched tenant holds the same
    /// prefix; select the least-loaded healthy matched tenant.
    fn select_matched_candidate(
        workers: &[Arc<dyn Worker>],
        healthy_indices: &[usize],
        result: &PrefixMatchResult,
    ) -> Option<usize> {
        let mut best: Option<usize> = None;
        let mut best_key: Option<(usize, usize, usize)> = None;
        for &idx in healthy_indices {
            let url = workers[idx].url();
            if result.matched_tenants.iter().any(|t| t.as_ref() == url) {
                let state = workers[idx].routing_state();
                let key = (state.load, state.processed, idx);
                match best_key {
                    Some(b) if key >= b => {}
                    _ => {
                        best = Some(idx);
                        best_key = Some(key);
                    }
                }
            }
        }
        best
    }

    /// Compute uncached prefill tokens by priority:
    /// 1. X-Prompt-Tokens header (exact).
    /// 2. (input_chars - matched_chars) / chars_per_token (char estimate).
    /// 3. None when neither is computable.
    fn compute_uncached_tokens(
        &self,
        info: &SelectWorkerInfo,
        result: &PrefixMatchResult,
    ) -> Option<usize> {
        // 1. Exact header value.
        if let Some(n) = parse_prompt_tokens_header(info.headers) {
            return Some(n);
        }
        // 2. Char-level estimate from the match result.
        let uncached_chars = result
            .input_char_count
            .saturating_sub(result.matched_char_count);
        if uncached_chars > 0 && self.config.chars_per_token > 0 {
            // Ceiling so a fractional block still counts as one token.
            let est = uncached_chars.div_ceil(self.config.chars_per_token);
            return Some(est);
        }
        None
    }

    /// Long request (uncached >= long_prefill_threshold). Does not overflow to
    /// a short-pool worker that already has load.
    fn select_long_request(
        &self,
        workers: &[Arc<dyn Worker>],
        long_indices: &[usize],
        short_indices: &[usize],
        _healthy_indices: &[usize],
        min_load_idx: Option<usize>,
    ) -> Option<usize> {
        let long_has_free = pool_has_free(workers, long_indices, self.config.long_pool_max_load);
        if long_has_free {
            return pool_min_load_worker(workers, long_indices);
        }
        // Long pool full/unhealthy: overflow to an idle short-pool worker only.
        if let Some(idx) = pool_idle_worker(workers, short_indices) {
            return Some(idx);
        }
        // Short pool all busy: queue on long pool if it still has a worker.
        if let Some(idx) = pool_min_load_worker(workers, long_indices) {
            return Some(idx);
        }
        // Long pool fully unhealthy and short pool busy: all-healthy min-load.
        min_load_idx
    }

    /// Short request (uncached < long_prefill_threshold). May overflow to an
    /// idle long-pool worker.
    fn select_short_request(
        &self,
        workers: &[Arc<dyn Worker>],
        long_indices: &[usize],
        short_indices: &[usize],
        _healthy_indices: &[usize],
        min_load_idx: Option<usize>,
    ) -> Option<usize> {
        let short_has_free = pool_has_free(workers, short_indices, self.config.short_pool_max_load);
        if short_has_free {
            return pool_min_load_worker(workers, short_indices);
        }
        // Short pool full: overflow to long pool if it has a free worker.
        let long_has_free = pool_has_free(workers, long_indices, self.config.long_pool_max_load);
        if long_has_free {
            return pool_min_load_worker(workers, long_indices);
        }
        // Both full: queue on short pool if it has a worker.
        if let Some(idx) = pool_min_load_worker(workers, short_indices) {
            return Some(idx);
        }
        // Short pool empty: queue on long pool if it has a worker.
        if let Some(idx) = pool_min_load_worker(workers, long_indices) {
            return Some(idx);
        }
        // Both pools empty: all-healthy min-load.
        min_load_idx
    }
}

/// Whether a worker belongs to the long pool (`labels["pool"] == "long"`).
fn is_long_pool(worker: &dyn Worker) -> bool {
    worker
        .metadata()
        .spec
        .labels
        .get("pool")
        .is_some_and(|v| v == "long")
}

/// Does any worker in `pool` have `load() < max_load`?
fn pool_has_free(workers: &[Arc<dyn Worker>], pool: &[usize], max_load: usize) -> bool {
    pool.iter()
        .any(|&i| workers[i].routing_state().load < max_load)
}

/// Return the index of an idle (`load == 0`) worker in `pool`, if any.
fn pool_idle_worker(workers: &[Arc<dyn Worker>], pool: &[usize]) -> Option<usize> {
    pool.iter()
        .copied()
        .find(|&i| workers[i].routing_state().load == 0)
}

/// Lowest-load worker in `pool` with the `(load, processed, idx)` tie-break.
/// Returns `None` when `pool` is empty.
fn pool_min_load_worker(workers: &[Arc<dyn Worker>], pool: &[usize]) -> Option<usize> {
    let mut best: Option<usize> = None;
    let mut best_key: Option<(usize, usize, usize)> = None;
    for &idx in pool {
        let state = workers[idx].routing_state();
        let key = (state.load, state.processed, idx);
        match best_key {
            Some(b) if key >= b => {}
            _ => {
                best = Some(idx);
                best_key = Some(key);
            }
        }
    }
    best
}

/// Parse the `X-Prompt-Tokens` header into a token count. Returns `None` on
/// missing/unparseable values. Header lookup is case-insensitive.
fn parse_prompt_tokens_header(headers: Option<&http::HeaderMap>) -> Option<usize> {
    let headers = headers?;
    headers
        .get(HEADER_PROMPT_TOKENS)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|n| *n > 0)
}

#[cfg(test)]
mod tests {
    use openai_protocol::worker::{HealthCheckConfig, WorkerStatus};

    use super::*;
    use crate::worker::{BasicWorkerBuilder, WorkerType};

    fn no_health_check() -> HealthCheckConfig {
        HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        }
    }

    /// Build a worker with an optional `pool` label and a pre-set load.
    fn make_worker(url: &str, pool: Option<&str>, load: usize) -> Arc<dyn Worker> {
        let mut builder = BasicWorkerBuilder::new(url)
            .worker_type(WorkerType::Regular)
            .api_key("test_api_key")
            .health_config(no_health_check());
        if let Some(p) = pool {
            builder = builder.label("pool", p);
        }
        let worker: Arc<dyn Worker> = Arc::new(builder.build());
        for _ in 0..load {
            // Leak the guard so the load stays elevated for the test without
            // holding a handle; the process tears down on test exit.
            std::mem::forget(crate::worker::WorkerLoadGuard::new(
                Arc::clone(&worker),
                None,
            ));
        }
        worker
    }

    fn info_with_text(text: &str) -> SelectWorkerInfo<'_> {
        SelectWorkerInfo {
            request_text: Some(text),
            ..Default::default()
        }
    }

    /// Build a `SelectWorkerInfo` with an `X-Prompt-Tokens` header. The header
    /// map must outlive the returned info — callers hold it in the same scope.
    fn info_with_header<'a>(headers: &'a http::HeaderMap, text: &'a str) -> SelectWorkerInfo<'a> {
        SelectWorkerInfo {
            request_text: Some(text),
            headers: Some(headers),
            ..Default::default()
        }
    }

    fn tokens_headers(tokens: usize) -> http::HeaderMap {
        let mut headers = http::HeaderMap::new();
        headers.insert(
            HEADER_PROMPT_TOKENS,
            http::HeaderValue::from_str(&tokens.to_string()).unwrap(),
        );
        headers
    }

    fn test_config() -> CacheAwareLengthConfig {
        CacheAwareLengthConfig {
            eviction_interval_secs: 0, // disable eviction thread in tests
            long_prefill_threshold: 100_000,
            long_pool_max_load: 2,
            short_pool_max_load: 2,
            chars_per_token: 4,
            ..Default::default()
        }
    }

    #[test]
    fn step1_returns_none_when_all_unhealthy() {
        let policy = CacheAwareLengthPolicy::with_config(test_config());
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("k")
                    .health_config(no_health_check())
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("k")
                    .health_config(no_health_check())
                    .build(),
            ),
        ];
        for w in &workers {
            w.set_status(WorkerStatus::NotReady);
        }
        assert!(policy
            .select_worker(&workers, &info_with_text("hello"))
            .is_none());
    }

    #[test]
    fn step3_cache_hit_pins_to_same_worker() {
        let policy = CacheAwareLengthPolicy::with_config(test_config());
        let workers: Vec<Arc<dyn Worker>> = vec![
            make_worker("http://w1:8000", None, 0),
            make_worker("http://w2:8000", None, 0),
        ];
        policy.init_workers(&workers);

        let prompt = "shared long prompt prefix that both workers could cache";
        let idx1 = policy
            .select_worker(&workers, &info_with_text(prompt))
            .unwrap();
        // Same prompt again → cache hit, same worker.
        let idx2 = policy
            .select_worker(&workers, &info_with_text(prompt))
            .unwrap();
        assert_eq!(idx1, idx2);
    }

    #[test]
    fn step3_tree_missing_falls_back_random() {
        // No init_workers → no tree for the model → random healthy, no panic.
        let policy = CacheAwareLengthPolicy::with_config(test_config());
        let workers: Vec<Arc<dyn Worker>> = vec![
            make_worker("http://w1:8000", None, 0),
            make_worker("http://w2:8000", None, 0),
        ];
        let idx = policy
            .select_worker(&workers, &info_with_text("novel prompt"))
            .unwrap();
        assert!(idx < workers.len());
    }

    #[test]
    fn step4_long_request_uses_long_pool_when_free() {
        let policy = CacheAwareLengthPolicy::with_config(test_config());
        let workers: Vec<Arc<dyn Worker>> = vec![
            make_worker("http://w1:8000", None, 0), // short pool, idle
            make_worker("http://w2:8000", Some("long"), 0), // long pool, free
        ];
        policy.init_workers(&workers);
        // novel prompt → no hit → long request via header.
        let headers = tokens_headers(200_000);
        let info = info_with_header(&headers, "novel prompt no match yet");
        let idx = policy.select_worker(&workers, &info).unwrap();
        assert_eq!(workers[idx].url(), "http://w2:8000");
    }

    #[test]
    fn step4_long_request_overflows_to_idle_short() {
        let policy = CacheAwareLengthPolicy::with_config(test_config());
        let workers: Vec<Arc<dyn Worker>> = vec![
            make_worker("http://w1:8000", None, 0), // short, idle (load 0)
            make_worker("http://w2:8000", Some("long"), 2), // long, full
        ];
        policy.init_workers(&workers);
        let headers = tokens_headers(200_000);
        let info = info_with_header(&headers, "novel prompt no match yet");
        let idx = policy.select_worker(&workers, &info).unwrap();
        assert_eq!(workers[idx].url(), "http://w1:8000");
    }

    #[test]
    fn step4_long_request_queues_on_long_when_short_busy() {
        let policy = CacheAwareLengthPolicy::with_config(test_config());
        let workers: Vec<Arc<dyn Worker>> = vec![
            make_worker("http://w1:8000", None, 1), // short, busy (load>0)
            make_worker("http://w2:8000", Some("long"), 2), // long, full but healthy
        ];
        policy.init_workers(&workers);
        let headers = tokens_headers(200_000);
        let info = info_with_header(&headers, "novel prompt no match yet");
        let idx = policy.select_worker(&workers, &info).unwrap();
        assert_eq!(workers[idx].url(), "http://w2:8000");
    }

    #[test]
    fn step4_short_request_uses_short_pool_when_free() {
        let policy = CacheAwareLengthPolicy::with_config(test_config());
        let workers: Vec<Arc<dyn Worker>> = vec![
            make_worker("http://w1:8000", None, 0),         // short, free
            make_worker("http://w2:8000", Some("long"), 0), // long, free
        ];
        policy.init_workers(&workers);
        let headers = tokens_headers(1_000); // short request
        let info = info_with_header(&headers, "novel prompt no match yet");
        let idx = policy.select_worker(&workers, &info).unwrap();
        assert_eq!(workers[idx].url(), "http://w1:8000");
    }

    #[test]
    fn step4_short_request_overflows_to_long_when_short_full() {
        let policy = CacheAwareLengthPolicy::with_config(test_config());
        let workers: Vec<Arc<dyn Worker>> = vec![
            make_worker("http://w1:8000", None, 2),         // short, full
            make_worker("http://w2:8000", Some("long"), 0), // long, free
        ];
        policy.init_workers(&workers);
        let headers = tokens_headers(1_000);
        let info = info_with_header(&headers, "novel prompt no match yet");
        let idx = policy.select_worker(&workers, &info).unwrap();
        assert_eq!(workers[idx].url(), "http://w2:8000");
    }

    #[test]
    fn step4_short_request_falls_back_to_short_when_both_full() {
        let policy = CacheAwareLengthPolicy::with_config(test_config());
        let workers: Vec<Arc<dyn Worker>> = vec![
            make_worker("http://w1:8000", None, 2),         // short, full
            make_worker("http://w2:8000", Some("long"), 2), // long, full
        ];
        policy.init_workers(&workers);
        let headers = tokens_headers(1_000);
        let info = info_with_header(&headers, "novel prompt no match yet");
        let idx = policy.select_worker(&workers, &info).unwrap();
        // short pool exists → queue on short pool min-load.
        assert_eq!(workers[idx].url(), "http://w1:8000");
    }

    #[test]
    fn step4_short_request_uses_long_when_short_pool_empty() {
        let policy = CacheAwareLengthPolicy::with_config(test_config());
        let workers: Vec<Arc<dyn Worker>> = vec![make_worker("http://w2:8000", Some("long"), 0)]; // only long
        policy.init_workers(&workers);
        let headers = tokens_headers(1_000);
        let info = info_with_header(&headers, "novel prompt no match yet");
        let idx = policy.select_worker(&workers, &info).unwrap();
        assert_eq!(workers[idx].url(), "http://w2:8000");
    }

    #[test]
    fn char_estimate_falls_back_when_no_header() {
        // No header: uncached derived from (input - matched) / chars_per_token.
        // A novel prompt with 400 chars / 4 = 100 tokens < threshold → short.
        let policy = CacheAwareLengthPolicy::with_config(test_config());
        let workers: Vec<Arc<dyn Worker>> = vec![
            make_worker("http://w1:8000", None, 0),
            make_worker("http://w2:8000", Some("long"), 0),
        ];
        policy.init_workers(&workers);
        let prompt = "a".repeat(400);
        let idx = policy
            .select_worker(&workers, &info_with_text(&prompt))
            .unwrap();
        assert_eq!(workers[idx].url(), "http://w1:8000");
    }

    #[test]
    fn is_long_pool_reads_label() {
        let w_long = make_worker("http://w:1", Some("long"), 0);
        let w_short = make_worker("http://w:2", None, 0);
        let w_other = make_worker("http://w:3", Some("short"), 0);
        assert!(is_long_pool(&*w_long));
        assert!(!is_long_pool(&*w_short));
        assert!(!is_long_pool(&*w_other));
    }

    #[test]
    fn parse_prompt_tokens_header_works() {
        let mut headers = http::HeaderMap::new();
        headers.insert(
            HEADER_PROMPT_TOKENS,
            http::HeaderValue::from_static("12345"),
        );
        assert_eq!(parse_prompt_tokens_header(Some(&headers)), Some(12345));
        assert_eq!(parse_prompt_tokens_header(None), None);
        headers.insert(
            HEADER_PROMPT_TOKENS,
            http::HeaderValue::from_static("notanum"),
        );
        assert_eq!(parse_prompt_tokens_header(Some(&headers)), None);
    }

    #[test]
    fn labels_map_is_accessible() {
        // Smoke-test the label access chain the policy depends on.
        let w = make_worker("http://w:1", Some("long"), 0);
        assert_eq!(
            w.metadata().spec.labels.get("pool").map(|s| s.as_str()),
            Some("long")
        );
    }
}
