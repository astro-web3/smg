/*
    Cache-Aware Length Load Balancing Router (cache_aware_length)

    A full superset of `cache_aware` that adds a long/short pool split on the
    no-cache (miss) branch. Uses a `NoCacheStrategy` trait injection: the
    inner `CacheAwarePolicy` handles all cache-affinity routing (string tree,
    token tree, event-driven, hash index, mesh sync, KV pressure), and when a
    request misses the cache, the `LengthStrategy` selects a worker based on
    uncached prefill tokens and the `pool` worker label.

    Pool membership:
        long pool  = healthy workers with labels["pool"] == "long"
        short pool = remaining healthy workers

    No-cache branch (Step 4):
        token source (priority):
          1. X-Prompt-Tokens header (exact, supplied by an upstream gateway).
          2. info.tokens length (token-only gRPC requests).
          3. (input_chars - matched_chars) / chars_per_token (char estimate).
          4. None computable → all-healthy min-load.
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

    This policy does NOT duplicate cache_aware's routing code — it holds an
    inner `CacheAwarePolicy` and delegates `select_worker` to it. The only
    customisation is the `NoCacheStrategy` implementation injected via
    `CacheAwarePolicy::with_no_cache_strategy`.
*/

use std::sync::Arc;

use tracing::debug;

use super::{
    normalize_model_key, CacheAwareConfig, CacheAwareLengthConfig, CacheAwarePolicy,
    LoadBalancingPolicy, NoCacheStrategy, SelectWorkerInfo,
};
use crate::{
    mesh::adapters::tree_sync::TreeSyncAdapter,
    worker::{KvEventMonitor, Worker},
};

use super::cache_aware::LoadReceiver;

/// HTTP header carrying the exact prompt token count, supplied by an
/// upstream gateway that has already tokenized the request.
const HEADER_PROMPT_TOKENS: &str = "x-prompt-tokens";

/// Cache-aware length routing policy — a full superset of `cache_aware`
/// that adds long/short pool split on the no-cache branch.
///
/// Holds an inner `CacheAwarePolicy` that owns the trees, mesh, KV monitor,
/// and all cache-affinity routing. The `LengthStrategy` (implementing
/// `NoCacheStrategy`) is injected into the inner policy so it intercepts
/// only the cache-miss branch.
#[derive(Debug)]
pub struct CacheAwareLengthPolicy {
    inner: CacheAwarePolicy,
    #[allow(dead_code)]
    config: CacheAwareLengthConfig,
    strategy: Arc<LengthStrategy>,
}

/// The no-cache strategy: selects a worker by splitting the healthy fleet
/// into long/short pools based on uncached prefill tokens and the `pool`
/// worker label.
#[derive(Debug)]
struct LengthStrategy {
    chars_per_token: usize,
    long_prefill_threshold: usize,
    long_pool_max_load: usize,
    short_pool_max_load: usize,
}

impl Default for CacheAwareLengthPolicy {
    fn default() -> Self {
        Self::with_config(CacheAwareLengthConfig::default())
    }
}

impl CacheAwareLengthPolicy {
    pub fn new() -> Self {
        Self::with_config(CacheAwareLengthConfig::default())
    }

    pub fn with_config(config: CacheAwareLengthConfig) -> Self {
        let strategy = Arc::new(LengthStrategy {
            chars_per_token: config.chars_per_token,
            long_prefill_threshold: config.long_prefill_threshold,
            long_pool_max_load: config.long_pool_max_load,
            short_pool_max_load: config.short_pool_max_load,
        });
        let inner = CacheAwarePolicy::with_config(config.base.clone())
            .with_no_cache_strategy(Arc::clone(&strategy) as Arc<dyn NoCacheStrategy>);
        Self {
            inner,
            config,
            strategy,
        }
    }

    // --- Delegated setters (forward to inner CacheAwarePolicy) ---

    pub fn set_kv_event_monitor(&self, monitor: Option<Arc<KvEventMonitor>>) {
        self.inner.set_kv_event_monitor(monitor);
    }

    pub fn set_load_receiver(&self, rx: Option<LoadReceiver>) {
        self.inner.set_load_receiver(rx);
    }

    pub fn set_mesh_tree_sync(&self, adapter: Option<Arc<TreeSyncAdapter>>) {
        self.inner.set_mesh_tree_sync(adapter);
    }

    pub fn init_workers(&self, workers: &[Arc<dyn Worker>]) {
        self.inner.init_workers(workers);
    }

    pub fn add_worker(&self, worker: &dyn Worker) {
        self.inner.add_worker(worker);
    }

    pub fn remove_worker_by_url(&self, url: &str) {
        self.inner.remove_worker_by_url(url);
    }

    /// Test-only access to the config so factory tests can verify values.
    #[cfg(test)]
    pub(crate) fn config_for_test(&self) -> &CacheAwareLengthConfig {
        &self.config
    }
}

impl LoadBalancingPolicy for CacheAwareLengthPolicy {
    fn select_worker(&self, workers: &[Arc<dyn Worker>], info: &SelectWorkerInfo) -> Option<usize> {
        // Delegate entirely to the inner CacheAwarePolicy. The LengthStrategy
        // (injected via with_no_cache_strategy) intercepts only the no-cache
        // branch; cache-hit, KV-pressure, hash-mode, and event-driven paths
        // run unchanged inside the inner policy.
        self.inner.select_worker(workers, info)
    }

    fn name(&self) -> &'static str {
        "cache_aware_length"
    }

    fn needs_request_text(&self) -> bool {
        true
    }

    fn needs_backend_loads(&self) -> bool {
        self.inner.needs_backend_loads()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl NoCacheStrategy for LengthStrategy {
    fn select_no_cache(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo,
        healthy_indices: &[usize],
        min_load_idx: Option<usize>,
        _avg_load: f64,
        _model_id: &str,
    ) -> Option<usize> {
        if healthy_indices.is_empty() {
            return min_load_idx;
        }

        // Compute uncached prefill tokens by priority:
        // 1. X-Prompt-Tokens header (exact).
        // 2. info.tokens length (token-only gRPC).
        // 3. (input_chars - matched_chars) / chars_per_token (char estimate).
        // 4. None → all-healthy min-load.
        let uncached_tokens = self.compute_uncached_tokens(info, workers, healthy_indices);

        let Some(uncached) = uncached_tokens else {
            // Neither source computable → all-healthy min-load.
            return min_load_idx;
        };

        // Split healthy workers into long/short pools by label.
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

        let selected = if uncached >= self.long_prefill_threshold {
            self.select_long_request(workers, &long_indices, &short_indices, min_load_idx)
        } else {
            self.select_short_request(workers, &long_indices, &short_indices, min_load_idx)
        };

        // The inner policy's caller (select_worker_min_load, the tree
        // closures, or hash_min_load) handles tree update +
        // increment_processed for the returned index.
        selected.or(min_load_idx)
    }
}

impl LengthStrategy {
    /// Compute uncached prefill tokens by priority.
    fn compute_uncached_tokens(
        &self,
        info: &SelectWorkerInfo,
        _workers: &[Arc<dyn Worker>],
        _healthy_indices: &[usize],
    ) -> Option<usize> {
        // 1. Exact header value.
        if let Some(n) = parse_prompt_tokens_header(info.headers) {
            return Some(n);
        }
        // 2. Token-only request (no text): use the token count directly.
        if let Some(tokens) = info.tokens {
            if !tokens.is_empty() {
                return Some(tokens.len());
            }
        }
        // 3. Char-level estimate from the request text.
        if let Some(text) = info.request_text {
            let input_chars = text.chars().count();
            if input_chars > 0 && self.chars_per_token > 0 {
                return Some(input_chars.div_ceil(self.chars_per_token));
            }
        }
        None
    }

    /// Long request (uncached >= long_prefill_threshold).
    fn select_long_request(
        &self,
        workers: &[Arc<dyn Worker>],
        long_indices: &[usize],
        short_indices: &[usize],
        min_load_idx: Option<usize>,
    ) -> Option<usize> {
        let long_has_free = pool_has_free(workers, long_indices, self.long_pool_max_load);
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

    /// Short request (uncached < long_prefill_threshold).
    fn select_short_request(
        &self,
        workers: &[Arc<dyn Worker>],
        long_indices: &[usize],
        short_indices: &[usize],
        min_load_idx: Option<usize>,
    ) -> Option<usize> {
        let short_has_free = pool_has_free(workers, short_indices, self.short_pool_max_load);
        if short_has_free {
            return pool_min_load_worker(workers, short_indices);
        }
        // Short pool full: overflow to long pool if it has a free worker.
        let long_has_free = pool_has_free(workers, long_indices, self.long_pool_max_load);
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
/// missing/unparseable values.
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
    use std::collections::HashMap;

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
            base: CacheAwareConfig {
                eviction_interval_secs: 0,
                ..Default::default()
            },
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
        let idx2 = policy
            .select_worker(&workers, &info_with_text(prompt))
            .unwrap();
        assert_eq!(idx1, idx2);
    }

    #[test]
    fn step3_tree_missing_falls_back_random() {
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
            make_worker("http://w1:8000", None, 0),
            make_worker("http://w2:8000", Some("long"), 0),
        ];
        policy.init_workers(&workers);
        let headers = tokens_headers(200_000);
        let info = info_with_header(&headers, "novel prompt no match yet");
        let idx = policy.select_worker(&workers, &info).unwrap();
        assert_eq!(workers[idx].url(), "http://w2:8000");
    }

    #[test]
    fn step4_long_request_overflows_to_idle_short() {
        let policy = CacheAwareLengthPolicy::with_config(test_config());
        let workers: Vec<Arc<dyn Worker>> = vec![
            make_worker("http://w1:8000", None, 0),
            make_worker("http://w2:8000", Some("long"), 2),
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
            make_worker("http://w1:8000", None, 1),
            make_worker("http://w2:8000", Some("long"), 2),
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
            make_worker("http://w1:8000", None, 0),
            make_worker("http://w2:8000", Some("long"), 0),
        ];
        policy.init_workers(&workers);
        let headers = tokens_headers(1_000);
        let info = info_with_header(&headers, "novel prompt no match yet");
        let idx = policy.select_worker(&workers, &info).unwrap();
        assert_eq!(workers[idx].url(), "http://w1:8000");
    }

    #[test]
    fn step4_short_request_overflows_to_long_when_short_full() {
        let policy = CacheAwareLengthPolicy::with_config(test_config());
        let workers: Vec<Arc<dyn Worker>> = vec![
            make_worker("http://w1:8000", None, 2),
            make_worker("http://w2:8000", Some("long"), 0),
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
            make_worker("http://w1:8000", None, 2),
            make_worker("http://w2:8000", Some("long"), 2),
        ];
        policy.init_workers(&workers);
        let headers = tokens_headers(1_000);
        let info = info_with_header(&headers, "novel prompt no match yet");
        let idx = policy.select_worker(&workers, &info).unwrap();
        assert_eq!(workers[idx].url(), "http://w1:8000");
    }

    #[test]
    fn step4_short_request_uses_long_when_short_pool_empty() {
        let policy = CacheAwareLengthPolicy::with_config(test_config());
        let workers: Vec<Arc<dyn Worker>> = vec![make_worker("http://w2:8000", Some("long"), 0)];
        policy.init_workers(&workers);
        let headers = tokens_headers(1_000);
        let info = info_with_header(&headers, "novel prompt no match yet");
        let idx = policy.select_worker(&workers, &info).unwrap();
        assert_eq!(workers[idx].url(), "http://w2:8000");
    }

    #[test]
    fn char_estimate_falls_back_when_no_header() {
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
    fn step3_hit_unhealthy_falls_back_to_first_healthy() {
        let policy = CacheAwareLengthPolicy::with_config(test_config());
        let workers: Vec<Arc<dyn Worker>> = vec![
            make_worker("http://w1:8000", None, 0),
            make_worker("http://w2:8000", Some("long"), 0),
        ];
        policy.init_workers(&workers);

        let prompt = "shared cache-building prompt for affinity";
        let first = policy
            .select_worker(&workers, &info_with_text(prompt))
            .unwrap();
        workers[first].set_status(WorkerStatus::NotReady);
        let second = policy
            .select_worker(&workers, &info_with_text(prompt))
            .unwrap();
        assert_ne!(workers[second].url(), workers[first].url());
    }

    #[test]
    fn step4_long_pool_unhealthy_overflows_to_idle_short() {
        let policy = CacheAwareLengthPolicy::with_config(test_config());
        let workers: Vec<Arc<dyn Worker>> = vec![
            make_worker("http://w1:8000", None, 0),
            make_worker("http://w2:8000", Some("long"), 0),
        ];
        policy.init_workers(&workers);
        workers[1].set_status(WorkerStatus::NotReady);
        let headers = tokens_headers(200_000);
        let info = info_with_header(&headers, "novel prompt no match yet");
        let idx = policy.select_worker(&workers, &info).unwrap();
        assert_eq!(workers[idx].url(), "http://w1:8000");
    }

    #[test]
    fn step4_header_overrides_char_estimate() {
        let policy = CacheAwareLengthPolicy::with_config(test_config());
        let workers: Vec<Arc<dyn Worker>> = vec![
            make_worker("http://w1:8000", None, 0),
            make_worker("http://w2:8000", Some("long"), 0),
        ];
        policy.init_workers(&workers);
        let mut headers = http::HeaderMap::new();
        headers.insert(
            HEADER_PROMPT_TOKENS,
            http::HeaderValue::from_static("200000"),
        );
        let info = SelectWorkerInfo {
            request_text: Some("short"),
            headers: Some(&headers),
            ..Default::default()
        };
        let idx = policy.select_worker(&workers, &info).unwrap();
        assert_eq!(workers[idx].url(), "http://w2:8000");
    }

    #[test]
    fn step4_token_only_request_uses_token_count() {
        let policy = CacheAwareLengthPolicy::with_config(test_config());
        let workers: Vec<Arc<dyn Worker>> = vec![
            make_worker("http://w1:8000", None, 0),
            make_worker("http://w2:8000", Some("long"), 0),
        ];
        policy.init_workers(&workers);
        let tokens: Vec<u32> = (0..200_000).collect();
        let info = SelectWorkerInfo {
            request_text: None,
            tokens: Some(&tokens),
            ..Default::default()
        };
        let idx = policy.select_worker(&workers, &info).unwrap();
        assert_eq!(workers[idx].url(), "http://w2:8000");
    }
}
