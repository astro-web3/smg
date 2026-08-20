use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::Instant,
};

use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
};
use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use openai_protocol::{
    chat::{ChatCompletionRequest, ChatMessage, MessageContent},
    common::{InputIds, StringOrArray},
    completion::CompletionRequest,
    generate::GenerateRequest,
    rerank::RerankRequest,
};
use reqwest::Client;
use serde::Serialize;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{debug, error, warn};

use crate::{
    config::types::RetryConfig,
    middleware::TenantRequestMeta,
    observability::{
        events::{self, Event},
        metrics::{bool_to_static_str, metrics_labels, Metrics},
        otel_trace::inject_trace_context_http,
        pd_request_lifecycle::{
            PdLifecycleConfig, PdRequestLifecycle, PdResponseAttribution, PdSseParser,
            PdStreamCancellationGuard, PdTerminalOutcome,
        },
    },
    policies::{LoadBalancingPolicy, PolicyRegistry, SelectWorkerInfo},
    routers::{
        common::{
            header_utils,
            retry::{is_retryable_status, RetryExecutor},
            sse::SseEncoder,
        },
        error,
        grpc::utils::{error_type_from_status, route_to_endpoint},
        RouterTrait,
    },
    worker::{HashRing, Worker, WorkerLoadGuard, WorkerRegistry, WorkerType, UNKNOWN_MODEL_ID},
};

const MAX_PD_SSE_EVENTS_PER_CHUNK: usize = 128;

#[derive(Debug)]
pub struct PDRouter {
    pub worker_registry: Arc<WorkerRegistry>,
    pub policy_registry: Arc<PolicyRegistry>,
    pub client: Client,
    pub retry_config: RetryConfig,
    pub api_key: Option<String>,
    pub pd_lifecycle_config: Option<Arc<PdLifecycleConfig>>,
}

#[derive(Clone)]
struct PDRequestContext<'a> {
    route: &'static str,
    batch_size: Option<usize>,
    is_stream: bool,
    return_logprob: bool,
    request_text: Option<String>,
    model_id: &'a str,
    headers: Option<HeaderMap>,
}

enum PdRelayEvent {
    Data {
        result: Result<Bytes, String>,
        observed_events: Vec<Value>,
        observation_overflowed: bool,
    },
    Terminal(PdTerminalOutcome),
}

/// Defers the lifecycle terminal transition until the downstream HTTP body
/// consumes the relay's terminal marker. Upstream completion alone is not
/// enough to claim that the Router finished serving the response body.
struct PdLifecycleRelayStream {
    receiver: ReceiverStream<PdRelayEvent>,
    request: Option<Arc<PdRequestLifecycle>>,
    status: StatusCode,
    terminal: bool,
}

impl PdLifecycleRelayStream {
    fn new(
        receiver: mpsc::Receiver<PdRelayEvent>,
        request: Option<Arc<PdRequestLifecycle>>,
        status: StatusCode,
    ) -> Self {
        Self {
            receiver: ReceiverStream::new(receiver),
            request,
            status,
            terminal: false,
        }
    }

    fn finish(&mut self, outcome: PdTerminalOutcome) {
        if let Some(request) = self.request.take() {
            request.finish_stream(outcome, self.status);
        }
        self.terminal = true;
    }
}

impl Stream for PdLifecycleRelayStream {
    type Item = Result<Bytes, String>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.terminal {
            return Poll::Ready(None);
        }

        match Pin::new(&mut self.receiver).poll_next(cx) {
            Poll::Ready(Some(PdRelayEvent::Data {
                result,
                observed_events,
                observation_overflowed,
            })) => {
                if let Some(request) = self.request.as_ref() {
                    if observation_overflowed {
                        request.mark_observation_overflowed();
                    }
                    for value in observed_events {
                        request.observe_json(&value);
                    }
                }
                if result.is_err() {
                    self.finish(PdTerminalOutcome::Error);
                }
                Poll::Ready(Some(result))
            }
            Poll::Ready(Some(PdRelayEvent::Terminal(outcome))) => {
                self.finish(outcome);
                Poll::Ready(None)
            }
            Poll::Ready(None) => {
                // A relay task that exits without its terminal marker is an
                // internal partial-stream failure, never success.
                self.finish(PdTerminalOutcome::Error);
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl PDRouter {
    fn finalize_pd_response(
        response: &mut Response,
        response_attribution: Option<PdResponseAttribution>,
    ) {
        // Worker-controlled headers must never cross a public PD response
        // boundary. The internal probe handler reads the trusted extension,
        // strips once more defensively, and then writes server-owned values.
        PdResponseAttribution::strip_internal_probe_headers(response.headers_mut());
        if let Some(attribution) = response_attribution {
            response.extensions_mut().insert(attribution);
        }
    }

    async fn proxy_to_first_prefill_worker(
        &self,
        endpoint: &str,
        headers: Option<Vec<(String, String)>>,
    ) -> Response {
        let workers = self.worker_registry.get_prefill_workers();
        let first_worker_url = workers.first().map(|w| w.url().to_string());

        if let Some(worker_url) = first_worker_url {
            self.proxy_to_worker(worker_url, endpoint, headers).await
        } else {
            error::service_unavailable("no_prefill_servers", "No prefill servers available")
        }
    }

    async fn proxy_to_worker(
        &self,
        worker_url: String,
        endpoint: &str,
        headers: Option<Vec<(String, String)>>,
    ) -> Response {
        let url = format!("{worker_url}/{endpoint}");
        let mut request_builder = self.client.get(&url);

        if let Some(headers) = headers {
            for (name, value) in headers {
                request_builder = request_builder.header(name, value);
            }
        }

        match request_builder.send().await {
            Ok(res) if res.status().is_success() => {
                let response_headers = header_utils::preserve_response_headers(res.headers());

                match res.bytes().await {
                    Ok(body) => {
                        let mut response = Response::new(Body::from(body));
                        *response.status_mut() = StatusCode::OK;
                        *response.headers_mut() = response_headers;
                        Self::finalize_pd_response(&mut response, None);
                        response
                    }
                    Err(e) => {
                        error!("Failed to read response body: {}", e);
                        error::internal_error(
                            "read_response_body_failed",
                            format!("Failed to read response body: {e}"),
                        )
                    }
                }
            }
            Ok(res) => {
                let status = StatusCode::from_u16(res.status().as_u16())
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                // Use the status code to determine which error function to use
                match status {
                    StatusCode::BAD_REQUEST => error::bad_request(
                        "server_bad_request",
                        format!("Server returned status: {}", res.status()),
                    ),
                    StatusCode::NOT_FOUND => error::not_found(
                        "server_not_found",
                        format!("Server returned status: {}", res.status()),
                    ),
                    StatusCode::INTERNAL_SERVER_ERROR => error::internal_error(
                        "server_internal_error",
                        format!("Server returned status: {}", res.status()),
                    ),
                    StatusCode::SERVICE_UNAVAILABLE => error::service_unavailable(
                        "server_unavailable",
                        format!("Server returned status: {}", res.status()),
                    ),
                    StatusCode::BAD_GATEWAY => error::bad_gateway(
                        "server_bad_gateway",
                        format!("Server returned status: {}", res.status()),
                    ),
                    _ => error::internal_error(
                        "server_error",
                        format!("Server returned status: {}", res.status()),
                    ),
                }
            }
            Err(e) => {
                error!("Failed to proxy request server: {}", e);
                error::internal_error(
                    "proxy_request_failed",
                    format!("Failed to proxy request: {e}"),
                )
            }
        }
    }

    #[expect(
        clippy::unused_async,
        reason = "async for API consistency with other router constructors"
    )]
    pub async fn new(ctx: &Arc<crate::app_context::AppContext>) -> Result<Self, String> {
        let pd_lifecycle_config = PdLifecycleConfig::from_env()?.map(Arc::new);
        if let Some(config) = pd_lifecycle_config.as_ref() {
            config.install_capability_anchor();
        }
        Ok(PDRouter {
            worker_registry: Arc::clone(&ctx.worker_registry),
            policy_registry: Arc::clone(&ctx.policy_registry),
            client: ctx.client.clone(),
            retry_config: ctx.router_config.effective_retry_config(),
            api_key: ctx.router_config.api_key.clone(),
            pd_lifecycle_config,
        })
    }

    fn handle_server_selection_error(error: String) -> Response {
        error!("Failed to select PD pair error={}", error);
        error::service_unavailable(
            "server_selection_failed",
            format!("No available servers: {error}"),
        )
    }

    fn handle_serialization_error(error: impl std::fmt::Display) -> Response {
        error!("Failed to serialize request error={}", error);
        error::internal_error("serialization_failed", "Failed to serialize request")
    }

    fn get_generate_batch_size(req: &GenerateRequest) -> Option<usize> {
        // GenerateRequest doesn't support batch via arrays, only via input_ids
        if let Some(InputIds::Batch(batches)) = &req.input_ids {
            if !batches.is_empty() {
                return Some(batches.len());
            }
        }
        None
    }

    fn get_chat_batch_size(req: &ChatCompletionRequest) -> Option<usize> {
        if let Some(n) = req.n {
            if n > 1 {
                return Some(n as usize);
            }
        }
        None
    }

    fn get_completion_batch_size(req: &CompletionRequest) -> Option<usize> {
        if let StringOrArray::Array(arr) = &req.prompt {
            if !arr.is_empty() {
                return Some(arr.len());
            }
        }
        None
    }

    // Static key strings to avoid per-request allocations
    const BOOTSTRAP_HOST_KEY: &'static str = "bootstrap_host";
    const BOOTSTRAP_PORT_KEY: &'static str = "bootstrap_port";
    const BOOTSTRAP_ROOM_KEY: &'static str = "bootstrap_room";

    fn inject_bootstrap_into_value(
        mut original: Value,
        prefill_worker: &dyn Worker,
        batch_size: Option<usize>,
    ) -> Result<Value, String> {
        let obj = original
            .as_object_mut()
            .ok_or_else(|| "Request must be a JSON object".to_string())?;

        if let Some(n) = batch_size {
            let mut hosts = Vec::with_capacity(n);
            let mut ports = Vec::with_capacity(n);
            let mut rooms = Vec::with_capacity(n);
            for _ in 0..n {
                hosts.push(prefill_worker.bootstrap_host());
                ports.push(prefill_worker.bootstrap_port());
                rooms.push(super::pd_types::generate_room_id());
            }
            obj.insert(
                Self::BOOTSTRAP_HOST_KEY.to_string(),
                Value::Array(hosts.into_iter().map(Value::from).collect()),
            );
            obj.insert(
                Self::BOOTSTRAP_PORT_KEY.to_string(),
                Value::Array(
                    ports
                        .into_iter()
                        .map(|p| match p {
                            Some(v) => Value::from(v),
                            None => Value::Null,
                        })
                        .collect(),
                ),
            );
            obj.insert(
                Self::BOOTSTRAP_ROOM_KEY.to_string(),
                Value::Array(rooms.into_iter().map(Value::from).collect()),
            );
        } else {
            obj.insert(
                Self::BOOTSTRAP_HOST_KEY.to_string(),
                Value::from(prefill_worker.bootstrap_host()),
            );
            obj.insert(
                Self::BOOTSTRAP_PORT_KEY.to_string(),
                match prefill_worker.bootstrap_port() {
                    Some(v) => Value::from(v),
                    None => Value::Null,
                },
            );
            obj.insert(
                Self::BOOTSTRAP_ROOM_KEY.to_string(),
                Value::from(super::pd_types::generate_room_id()),
            );
        }
        Ok(original)
    }

    fn inject_dp_rank_to_json(json_val: &mut Value, rank: isize, rank_key: &str) {
        if let Some(obj) = json_val.as_object_mut() {
            obj.insert(rank_key.to_string(), Value::Number(rank.into()));
        }
    }

    async fn execute_dual_dispatch<T: Serialize + Clone>(
        &self,
        headers: Option<&HeaderMap>,
        request_meta: &TenantRequestMeta,
        original_request: &T,
        context: PDRequestContext<'_>,
    ) -> Response {
        let start_time = Instant::now();

        let route = context.route;
        let model = context.model_id;
        let endpoint = route_to_endpoint(route);
        let is_stream = context.is_stream;
        let request_lifecycle = if let Some(config) = self.pd_lifecycle_config.as_ref() {
            let classification =
                match config.classify_public_http_request(request_meta, context.route, headers) {
                    Ok(classification) => classification,
                    Err(message) => {
                        return error::create_error(
                            StatusCode::FORBIDDEN,
                            "pd_request_class_rejected",
                            message,
                        );
                    }
                };
            Some(PdRequestLifecycle::start_classified(
                Arc::clone(config),
                classification,
            ))
        } else {
            None
        };
        let response_attribution = request_lifecycle
            .as_ref()
            .and_then(|lifecycle| lifecycle.response_attribution())
            .cloned();
        let retry_lifecycle = request_lifecycle.clone();

        // Record request start (Layer 2)
        Metrics::record_router_request(
            metrics_labels::ROUTER_HTTP,
            metrics_labels::BACKEND_PD,
            metrics_labels::CONNECTION_HTTP,
            model,
            endpoint,
            bool_to_static_str(context.is_stream),
        );
        // Clone request once outside the retry loop, then use Arc to share across attempts
        // This avoids O(retries) clones by sharing the same data
        let shared_request = Arc::new(original_request.clone());

        // Use per-model retry config if set by a worker, otherwise fall back to router default.
        let per_model_retry_config = self.worker_registry.get_retry_config(model);
        let retry_config = per_model_retry_config
            .as_ref()
            .unwrap_or(&self.retry_config);

        let mut response = RetryExecutor::execute_response_with_retry(
            retry_config,
            {
                move |attempt: u32| {
                    // Clone Arc (cheap reference count increment) instead of cloning the entire request
                    let shared_request = Arc::clone(&shared_request);
                    let context = context.clone();
                    let request_lifecycle = retry_lifecycle.clone();
                    async move {
                        if let Some(lifecycle) = request_lifecycle.as_ref() {
                            lifecycle.begin_unassigned_attempt();
                        }
                        let required_cache_domain = request_lifecycle
                            .as_ref()
                            .and_then(|lifecycle| lifecycle.response_attribution())
                            .map(|attribution| attribution.cache_domain());
                        let (prefill, decode) = match self
                            .select_pd_pair(
                                context.request_text.as_deref(),
                                context.model_id,
                                context.headers.as_ref(),
                                required_cache_domain,
                            )
                            .await
                        {
                            Ok(pair) => pair,
                            Err(e) => {
                                return Self::handle_server_selection_error(e);
                            }
                        };

                        if let Some(lifecycle) = request_lifecycle.as_ref() {
                            if let Err(message) =
                                lifecycle.begin_selected_attempt(prefill.as_ref(), decode.as_ref())
                            {
                                return error::create_error(
                                    StatusCode::BAD_GATEWAY,
                                    "pd_attribution_mismatch",
                                    message,
                                );
                            }
                        }

                        debug!(
                            "PD retry attempt {} using prefill={} decode={}",
                            attempt,
                            prefill.url(),
                            decode.url()
                        );

                        let mut json_request = match serde_json::to_value(shared_request.as_ref()) {
                            Ok(v) => v,
                            Err(e) => return Self::handle_serialization_error(e),
                        };

                        json_request = match Self::inject_bootstrap_into_value(
                            json_request,
                            prefill.as_ref(),
                            context.batch_size,
                        ) {
                            Ok(v) => v,
                            Err(e) => {
                                Metrics::record_pd_bootstrap_failure();
                                return Self::handle_serialization_error(e);
                            }
                        };

                        let mut prefill_json_request = json_request.clone();
                        let mut decode_json_request = json_request;

                        let mut prefill_rank = prefill.dp_rank().map(|rank| rank as isize);
                        let mut decode_rank = decode.dp_rank().map(|rank| rank as isize);

                        let dp_rank_policy_opt = self.policy_registry.get_dp_rank_policy();
                        if let Some(dp_rank_policy) = dp_rank_policy_opt.as_ref() {
                            let estimated_cost: isize = match context.request_text.as_ref() {
                                Some(text) => {
                                    // Calculate token count using a simple heuristic
                                    // In a real implementation, we would use the tokenizer
                                    // For now, use a simple words-to-tokens ratio
                                    let word_count = text.split_whitespace().count();
                                    // Assume average 1.3 tokens per word
                                    let token_count = (word_count as f64 * 1.3).ceil() as isize;
                                    token_count.max(1)
                                }
                                None => 1, // Use at least 1 to avoid no-op
                            };
                            let policy_prefill_rank =
                                dp_rank_policy.select_dp_rank(prefill.as_ref(), estimated_cost);
                            let policy_decode_rank =
                                dp_rank_policy.select_dp_rank(decode.as_ref(), estimated_cost);
                            if let Some(rank) = policy_prefill_rank {
                                prefill_rank = Some(rank);
                            }
                            if let Some(rank) = policy_decode_rank {
                                decode_rank = Some(rank);
                            }
                        }

                        if let Some(p_rank) = prefill_rank {
                            Self::inject_dp_rank_to_json(
                                &mut prefill_json_request,
                                p_rank,
                                "routed_dp_rank",
                            );
                            Self::inject_dp_rank_to_json(
                                &mut decode_json_request,
                                p_rank,
                                "disagg_prefill_dp_rank",
                            );
                        }
                        if let Some(d_rank) = decode_rank {
                            Self::inject_dp_rank_to_json(
                                &mut decode_json_request,
                                d_rank,
                                "routed_dp_rank",
                            );
                        }
                        if prefill_rank.is_some() || decode_rank.is_some() {
                            debug!(
                                "PD selected DP ranks prefill={:?} decode={:?}",
                                prefill_rank, decode_rank
                            );
                        }

                        let response = self
                            .execute_dual_dispatch_internal(
                                headers,
                                prefill_json_request,
                                decode_json_request,
                                context,
                                Arc::clone(&prefill),
                                Arc::clone(&decode),
                                request_lifecycle,
                            )
                            .await;

                        let status = response.status();
                        prefill.record_outcome(status.as_u16());
                        decode.record_outcome(status.as_u16());

                        // Record worker errors for server errors (5xx)
                        if status.is_server_error() {
                            let error_type = error_type_from_status(status);
                            Metrics::record_worker_error(
                                metrics_labels::WORKER_PREFILL,
                                metrics_labels::CONNECTION_HTTP,
                                error_type,
                            );
                            Metrics::record_worker_error(
                                metrics_labels::WORKER_DECODE,
                                metrics_labels::CONNECTION_HTTP,
                                error_type,
                            );
                        }

                        response
                    }
                }
            },
            |res, _attempt| is_retryable_status(res.status()),
            |delay, attempt| {
                // Layer 3 worker metrics (PD mode uses both prefill and decode workers)
                Metrics::record_worker_retry(metrics_labels::WORKER_PREFILL, endpoint);
                Metrics::record_worker_retry(metrics_labels::WORKER_DECODE, endpoint);
                Metrics::record_worker_retry_backoff(attempt, delay);
            },
            || {
                Metrics::record_worker_retries_exhausted(metrics_labels::WORKER_PREFILL, endpoint);
                Metrics::record_worker_retries_exhausted(metrics_labels::WORKER_DECODE, endpoint);
            },
        )
        .await;

        // Record Layer 2 metrics
        let duration = start_time.elapsed();
        if response.status().is_success() {
            Metrics::record_router_duration(
                metrics_labels::ROUTER_HTTP,
                metrics_labels::BACKEND_PD,
                metrics_labels::CONNECTION_HTTP,
                model,
                endpoint,
                duration,
            );
        } else if !is_retryable_status(response.status()) {
            Metrics::record_router_error(
                metrics_labels::ROUTER_HTTP,
                metrics_labels::BACKEND_PD,
                metrics_labels::CONNECTION_HTTP,
                model,
                endpoint,
                error_type_from_status(response.status()),
            );
        }

        // Successful streams finish in the body relay, where completion,
        // partial-stream failure, and client cancellation are observable. All
        // other responses are complete here after the final retry attempt.
        if let Some(lifecycle) = request_lifecycle {
            if !is_stream || !response.status().is_success() {
                let outcome = if response.status().is_success() {
                    PdTerminalOutcome::Success
                } else {
                    PdTerminalOutcome::Error
                };
                lifecycle.finish_stream(outcome, response.status());
            }
        }

        // Keep attribution server-side while stripping every worker-supplied
        // internal header at the shared PD response boundary.
        Self::finalize_pd_response(&mut response, response_attribution);

        response
    }

    async fn handle_decode_error_response(
        &self,
        res: reqwest::Response,
        context: &PDRequestContext<'_>,
        decode: Arc<dyn Worker>,
        load_guards: Vec<WorkerLoadGuard>,
    ) -> Response {
        let status = res.status();

        if context.is_stream {
            // Handle streaming error response
            let response_headers = header_utils::preserve_response_headers(res.headers());
            let error_payload = match res.bytes().await {
                Ok(error_body) => {
                    if let Ok(error_json) = serde_json::from_slice::<Value>(&error_body) {
                        json!({ "message": error_json, "status": status.as_u16() })
                    } else {
                        json!({ "message": String::from_utf8_lossy(&error_body).to_string(), "status": status.as_u16() })
                    }
                }
                Err(e) => {
                    json!({ "message": format!("Decode server error: {}", e), "status": status.as_u16() })
                }
            };

            let sse_data = format!(
                "data: {}\n\n",
                serde_json::to_string(&json!({ "error": error_payload })).unwrap_or_default()
            );
            let error_stream = tokio_stream::once(Ok(Bytes::from(sse_data)));

            let decode_url = decode.url().to_string();
            self.create_streaming_response(
                error_stream,
                status,
                None,
                context.return_logprob,
                Some(decode_url),
                Some(response_headers),
                load_guards,
                None,
            )
        } else {
            // Handle non-streaming error response
            match res.bytes().await {
                Ok(error_body) => {
                    // Try to parse error message from body, fallback to status-based error
                    let error_message = if let Ok(error_json) =
                        serde_json::from_slice::<Value>(&error_body)
                    {
                        if let Some(msg) = error_json
                            .get("error")
                            .and_then(|e| e.get("message"))
                            .and_then(|m| m.as_str())
                        {
                            msg.to_string()
                        } else if let Some(msg) = error_json.get("message").and_then(|m| m.as_str())
                        {
                            msg.to_string()
                        } else {
                            String::from_utf8_lossy(&error_body).to_string()
                        }
                    } else {
                        String::from_utf8_lossy(&error_body).to_string()
                    };

                    let status_code = StatusCode::from_u16(status.as_u16())
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                    match status_code {
                        StatusCode::BAD_REQUEST => {
                            error::bad_request("decode_bad_request", error_message)
                        }
                        StatusCode::NOT_FOUND => {
                            error::not_found("decode_not_found", error_message)
                        }
                        StatusCode::INTERNAL_SERVER_ERROR => {
                            error::internal_error("decode_internal_error", error_message)
                        }
                        StatusCode::SERVICE_UNAVAILABLE => {
                            error::service_unavailable("decode_unavailable", error_message)
                        }
                        StatusCode::BAD_GATEWAY => {
                            error::bad_gateway("decode_bad_gateway", error_message)
                        }
                        _ => error::internal_error("decode_error", error_message),
                    }
                }
                Err(e) => {
                    let error_message = format!("Decode server error: {e}");
                    let status_code = StatusCode::from_u16(status.as_u16())
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                    match status_code {
                        StatusCode::BAD_REQUEST => {
                            error::bad_request("decode_read_failed", error_message)
                        }
                        StatusCode::NOT_FOUND => {
                            error::not_found("decode_read_failed", error_message)
                        }
                        StatusCode::INTERNAL_SERVER_ERROR => {
                            error::internal_error("decode_read_failed", error_message)
                        }
                        StatusCode::SERVICE_UNAVAILABLE => {
                            error::service_unavailable("decode_read_failed", error_message)
                        }
                        StatusCode::BAD_GATEWAY => {
                            error::bad_gateway("decode_read_failed", error_message)
                        }
                        _ => error::internal_error("decode_read_failed", error_message),
                    }
                }
            }
        }
    }

    // Internal method that performs the actual dual dispatch (without retry logic)
    #[expect(
        clippy::too_many_arguments,
        reason = "PD dispatch needs both leg requests, selected workers, and one lifecycle"
    )]
    async fn execute_dual_dispatch_internal(
        &self,
        headers: Option<&HeaderMap>,
        prefill_json_request: Value,
        decode_json_request: Value,
        context: PDRequestContext<'_>,
        prefill: Arc<dyn Worker>,
        decode: Arc<dyn Worker>,
        request_lifecycle: Option<Arc<PdRequestLifecycle>>,
    ) -> Response {
        let load_guards = vec![
            WorkerLoadGuard::new(prefill.clone(), headers),
            WorkerLoadGuard::new(decode.clone(), headers),
        ];

        let mut headers_with_trace = headers.cloned().unwrap_or_default();
        inject_trace_context_http(&mut headers_with_trace);
        let headers = Some(&headers_with_trace);

        // Build both requests
        let prefill_request = self.build_post_with_headers(
            &self.client,
            prefill.as_ref(),
            context.route,
            &prefill_json_request,
            headers,
            false,
        );
        let decode_request = self.build_post_with_headers(
            &self.client,
            decode.as_ref(),
            context.route,
            &decode_json_request,
            headers,
            false,
        );

        // Send both requests concurrently and wait for both
        // Note: Using borrowed references avoids heap allocation
        events::RequestPDSentEvent {
            prefill_url: prefill.url(),
            decode_url: decode.url(),
        }
        .emit();

        // Send both requests concurrently. Use try_join so that if either side
        // hits a transport error, the other is cancelled immediately — otherwise
        // the surviving request hangs waiting for a PD bootstrap that will never
        // come (see #831).
        // Each leg captures its own head-arrival elapsed when its `send()`
        // resolves, so the two are independent even though `try_join!` returns
        // only once both heads arrive: decode TTFT isn't conflated with the
        // prefill-head wait, and prefill duration isn't conflated with a slower
        // decode head. Recorded on the success path only.
        let runtime = prefill.metadata().spec.runtime_type.as_str();
        let dispatch_start = Instant::now();
        let prefill_fut = async {
            let resp = prefill_request.send().await?;
            Ok::<_, reqwest::Error>((dispatch_start.elapsed(), resp))
        };
        let decode_fut = async {
            let resp = decode_request.send().await?;
            Ok::<_, reqwest::Error>((dispatch_start.elapsed(), resp))
        };
        let pd_result = tokio::try_join!(prefill_fut, decode_fut);

        events::RequestReceivedEvent {}.emit();

        let ((prefill_head_elapsed, prefill_response), (decode_head_elapsed, decode_response)) =
            match pd_result {
                Ok(pair) => pair,
                Err(e) => {
                    error!("PD request transport error, both sides aborted: {e}");
                    // Don't record_outcome here — the caller (execute_dual_dispatch)
                    // records outcomes from the response status after we return.
                    return error::bad_gateway(
                        "PD disaggregation request failed",
                        format!("Transport error: {e}"),
                    );
                }
            };

        // Process decode response
        let status = StatusCode::from_u16(decode_response.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        debug!("Decode response status: {}", status);

        if !status.is_success() {
            error!(
                "Decode server returned error status decode_url={} status={}",
                decode.url(),
                status
            );

            return self
                .handle_decode_error_response(decode_response, &context, decode, load_guards)
                .await;
        }

        // Honest PD TTFT: dispatch to the decode response head — the first
        // user-visible decode output, since the gateway forwards the decode body
        // unbuffered. Complements the decode-only `smg_router_ttft_seconds`,
        // which PD never narrows to a single leg.
        Metrics::record_pd_ttft(
            metrics_labels::BACKEND_PD,
            context.model_id,
            runtime,
            decode_head_elapsed,
        );

        // Process prefill response
        let prefill_drain_start = Instant::now();
        let prefill_body = match self
            .process_prefill_response(prefill_response, prefill.url(), context.return_logprob)
            .await
        {
            Ok((_, body)) => body,
            Err(error_response) => return error_response,
        };

        // Prefill RPC duration: prefill-head elapsed + body drain, independent
        // of decode so a slower decode head never inflates it.
        Metrics::record_pd_prefill_duration(
            metrics_labels::BACKEND_PD,
            context.model_id,
            runtime,
            prefill_head_elapsed + prefill_drain_start.elapsed(),
        );

        if context.is_stream {
            // Streaming response
            let prefill_logprobs = if context.return_logprob {
                prefill_body
                    .as_ref()
                    .and_then(|body| serde_json::from_slice::<Value>(body).ok())
                    .and_then(|json| json.pointer("/meta_info/input_token_logprobs").cloned())
            } else {
                None
            };

            let response_headers =
                header_utils::preserve_response_headers(decode_response.headers());

            self.create_streaming_response(
                decode_response.bytes_stream(),
                status,
                prefill_logprobs,
                context.return_logprob,
                None,
                Some(response_headers),
                load_guards,
                request_lifecycle,
            )
        } else {
            // Non-streaming response
            if context.return_logprob {
                self.process_non_streaming_response(
                    decode_response,
                    status,
                    context.return_logprob,
                    prefill_body,
                    request_lifecycle.as_ref(),
                )
                .await
            } else {
                // Direct passthrough when no logprobs needed
                let response_headers =
                    header_utils::preserve_response_headers(decode_response.headers());

                match decode_response.bytes().await {
                    Ok(decode_body) => {
                        Self::observe_pd_response_body(request_lifecycle.as_ref(), &decode_body);
                        let mut response = Response::new(Body::from(decode_body));
                        *response.status_mut() = status;
                        *response.headers_mut() = response_headers;
                        response
                    }
                    Err(e) => {
                        error!("Failed to read decode response: {}", e);
                        error::internal_error("read_response_failed", "Failed to read response")
                    }
                }
            }
        }
    }

    fn policies_need_request_text(&self) -> bool {
        let prefill_policy = self.policy_registry.get_prefill_policy();
        let decode_policy = self.policy_registry.get_decode_policy();
        prefill_policy.needs_request_text() || decode_policy.needs_request_text()
    }

    #[expect(
        clippy::unused_async,
        reason = "async for API consistency; callers await uniformly"
    )]
    async fn select_pd_pair(
        &self,
        request_text: Option<&str>,
        model_id: &str,
        headers: Option<&HeaderMap>,
        required_cache_domain: Option<&str>,
    ) -> Result<(Arc<dyn Worker>, Arc<dyn Worker>), String> {
        debug!("Selecting PD pair: model_id={:?}", model_id);

        let is_unknown_model = model_id == UNKNOWN_MODEL_ID;

        let prefill_workers = {
            let by_model: Vec<_> = self
                .worker_registry
                .get_by_model(model_id)
                .iter()
                .filter(|w| matches!(w.worker_type(), WorkerType::Prefill))
                .cloned()
                .collect();
            let workers = if by_model.is_empty() && is_unknown_model {
                // "auto" means pick any — fall back to all prefill workers
                self.worker_registry.get_prefill_workers().to_vec()
            } else {
                by_model
            };
            workers
                .into_iter()
                .filter(|worker| {
                    required_cache_domain.is_none_or(|required| {
                        worker
                            .metadata()
                            .spec
                            .labels
                            .get("cache_domain")
                            .is_some_and(|actual| actual == required)
                    })
                })
                .collect::<Vec<_>>()
        };

        let decode_workers = {
            let by_model: Vec<_> = self
                .worker_registry
                .get_by_model(model_id)
                .iter()
                .filter(|w| matches!(w.worker_type(), WorkerType::Decode))
                .cloned()
                .collect();
            if by_model.is_empty() && is_unknown_model {
                // Only fall back to all workers when model is "unknown" (wildcard)
                self.worker_registry.get_decode_workers().to_vec()
            } else {
                by_model
            }
        };

        let prefill_policy = self.policy_registry.get_prefill_policy();
        let decode_policy = self.policy_registry.get_decode_policy();

        // Get cached hash ring for consistent hashing
        let hash_ring = self.worker_registry.get_hash_ring(model_id);

        let prefill = self.pick_worker_by_policy_arc(
            &prefill_workers,
            &prefill_policy,
            request_text,
            headers,
            hash_ring.clone(),
            "prefill",
            crate::policies::WorkerLeg::Prefill,
        )?;

        let decode_workers = if required_cache_domain.is_some() {
            let pair_id = prefill
                .metadata()
                .spec
                .labels
                .get("execution_cohort_pair_id")
                .ok_or_else(|| {
                    "Selected prefill worker is missing execution_cohort_pair_id".to_string()
                })?;
            decode_workers
                .into_iter()
                .filter(|worker| {
                    worker
                        .metadata()
                        .spec
                        .labels
                        .get("execution_cohort_pair_id")
                        == Some(pair_id)
                })
                .collect::<Vec<_>>()
        } else {
            decode_workers
        };

        let decode = self.pick_worker_by_policy_arc(
            &decode_workers,
            &decode_policy,
            request_text,
            headers,
            hash_ring,
            "decode",
            crate::policies::WorkerLeg::Decode,
        )?;

        // Record worker selection metrics (Layer 3)
        let model = model_id;
        Metrics::record_worker_selection(
            metrics_labels::WORKER_PREFILL,
            metrics_labels::CONNECTION_HTTP,
            model,
            prefill_policy.name(),
        );
        Metrics::record_worker_selection(
            metrics_labels::WORKER_DECODE,
            metrics_labels::CONNECTION_HTTP,
            model,
            decode_policy.name(),
        );

        Ok((prefill, decode))
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "HTTP PD worker pick threads policy + request context + leg"
    )]
    fn pick_worker_by_policy_arc(
        &self,
        workers: &[Arc<dyn Worker>],
        policy: &Arc<dyn LoadBalancingPolicy>,
        request_text: Option<&str>,
        headers: Option<&HeaderMap>,
        hash_ring: Option<Arc<HashRing>>,
        worker_type: &str,
        leg: crate::policies::WorkerLeg,
    ) -> Result<Arc<dyn Worker>, String> {
        if workers.is_empty() {
            return Err(format!(
                "No {worker_type} workers available. Please check if {worker_type} servers are configured and healthy."
            ));
        }

        let available_workers: Vec<Arc<dyn Worker>> = workers
            .iter()
            .filter(|w| w.is_available())
            .cloned()
            .collect();

        if available_workers.is_empty() {
            return Err(format!(
                "No available {worker_type} workers (all circuits open or unhealthy)"
            ));
        }

        let selected_idx = self
            .policy_registry
            .select_worker(
                policy,
                &available_workers,
                &SelectWorkerInfo {
                    request_text,
                    tokens: None, // HTTP doesn't have tokens, use gRPC for PrefixHash
                    headers,
                    hash_ring,
                    leg,
                },
            )
            .ok_or_else(|| {
                format!(
                    "Policy {} failed to select a {} worker",
                    policy.name(),
                    worker_type
                )
            })?;

        Ok(available_workers[selected_idx].clone())
    }

    #[expect(clippy::too_many_arguments)]
    #[expect(
        clippy::unused_self,
        reason = "method on PDRouter for consistent API; may use self in future"
    )]
    fn create_streaming_response(
        &self,
        stream: impl Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
        status: StatusCode,
        prefill_logprobs: Option<Value>,
        return_logprob: bool,
        decode_url: Option<String>,
        headers: Option<HeaderMap>,
        load_guards: Vec<WorkerLoadGuard>,
        request_lifecycle: Option<Arc<PdRequestLifecycle>>,
    ) -> Response {
        use crate::worker::AttachedBody;

        // Keep upstream reads close to downstream body consumption. Besides
        // bounding memory, this prevents a fast Decode from racing arbitrarily
        // far ahead of a slow or disconnected client.
        let (tx, rx) = mpsc::channel(8);
        let cancellation_guard = PdStreamCancellationGuard::new(request_lifecycle.clone(), status);
        let observe_lifecycle = request_lifecycle.is_some();
        let merge_logprobs = return_logprob && prefill_logprobs.is_some();

        #[expect(
            clippy::disallowed_methods,
            reason = "fire-and-forget stream relay; gateway shutdown need not wait for decode stream forwarding"
        )]
        tokio::spawn(async move {
            futures_util::pin_mut!(stream);
            let mut terminal_outcome = PdTerminalOutcome::Error;
            let mut sse_parser = PdSseParser::default();
            // Reusable SSE encoder for the logprob-merge re-encode path.
            let mut encoder = SseEncoder::new();
            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        let mut observed_events = Vec::new();
                        let mut observation_overflowed = false;
                        let event_was_pending = sse_parser.has_pending_event();
                        let parse_state = sse_parser.observe_chunk(&chunk, |value| {
                            if observe_lifecycle || merge_logprobs {
                                if observed_events.len() < MAX_PD_SSE_EVENTS_PER_CHUNK {
                                    observed_events.push(value);
                                } else {
                                    observation_overflowed = true;
                                }
                            }
                        });
                        observation_overflowed |= parse_state.buffer_overflowed;

                        let result = if merge_logprobs && !observation_overflowed {
                            let event = if !event_was_pending && observed_events.len() == 1 {
                                observed_events.first_mut()
                            } else {
                                None
                            };
                            Self::merge_streaming_logprobs(
                                prefill_logprobs.as_ref(),
                                event,
                                &mut encoder,
                            )
                            .unwrap_or(chunk)
                        } else {
                            chunk
                        };

                        if tx
                            .send(PdRelayEvent::Data {
                                result: Ok(result),
                                observed_events,
                                observation_overflowed,
                            })
                            .await
                            .is_err()
                        {
                            terminal_outcome = PdTerminalOutcome::Cancel;
                            break;
                        }

                        if parse_state.done {
                            terminal_outcome = if parse_state.application_error_seen {
                                PdTerminalOutcome::Error
                            } else {
                                PdTerminalOutcome::Success
                            };
                            break;
                        }
                    }
                    Err(e) => {
                        if let Some(ref url) = decode_url {
                            error!("Stream error from decode server {}: {}", url, e);
                        }
                        let _ = tx
                            .send(PdRelayEvent::Data {
                                result: Err(format!("Stream error: {e}")),
                                observed_events: Vec::new(),
                                observation_overflowed: false,
                            })
                            .await;
                        break;
                    }
                }
            }
            let _ = tx.send(PdRelayEvent::Terminal(terminal_outcome)).await;
        });

        let stream = PdLifecycleRelayStream::new(rx, request_lifecycle, status);
        let body = Body::from_stream(stream);

        let mut response = Response::new(body);
        *response.status_mut() = status;

        let mut response_headers = headers.unwrap_or_default();
        response_headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
        *response.headers_mut() = response_headers;

        AttachedBody::wrap_response(response, (load_guards, cancellation_guard))
    }

    /// Build a non-streaming PD response with `Content-Type: application/json`.
    ///
    /// Axum's `(StatusCode, Bytes).into_response()` defaults to
    /// `application/octet-stream`, which breaks OpenAI-style JSON clients.
    fn non_stream_pd_json_response(status: StatusCode, body: Bytes) -> Response {
        let mut response = Response::new(Body::from(body));
        *response.status_mut() = status;
        response
            .headers_mut()
            .insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        response
    }

    // Helper to process non-streaming decode response with logprob merging
    async fn process_non_streaming_response(
        &self,
        res: reqwest::Response,
        status: StatusCode,
        return_logprob: bool,
        prefill_body: Option<Bytes>,
        request_lifecycle: Option<&Arc<PdRequestLifecycle>>,
    ) -> Response {
        let response = res.bytes().await;
        let decode_body = match response {
            Ok(decode_body) => decode_body,
            Err(e) => {
                error!("Failed to read decode response: {}", e);
                return error::internal_error("read_response_failed", "Failed to read response");
            }
        };
        Self::observe_pd_response_body(request_lifecycle, &decode_body);

        if !return_logprob {
            return Self::non_stream_pd_json_response(status, decode_body);
        }

        let Some(prefill_body) = prefill_body else {
            return Self::non_stream_pd_json_response(status, decode_body);
        };

        // Merge logprobs from prefill and decode
        let (Ok(prefill_json), Ok(mut decode_json)) = (
            serde_json::from_slice::<Value>(&prefill_body),
            serde_json::from_slice::<Value>(&decode_body),
        ) else {
            warn!("Failed to parse responses for logprob merging");
            return Self::non_stream_pd_json_response(status, decode_body);
        };

        Self::merge_logprobs_in_json(&prefill_json, &mut decode_json);

        // Return merged response
        match serde_json::to_vec(&decode_json) {
            Ok(body) => Self::non_stream_pd_json_response(status, Bytes::from(body)),
            Err(e) => {
                error!("Failed to serialize merged response: {}", e);
                Self::non_stream_pd_json_response(status, decode_body)
            }
        }
    }

    fn observe_pd_response_body(request_lifecycle: Option<&Arc<PdRequestLifecycle>>, body: &[u8]) {
        if let (Some(lifecycle), Ok(value)) =
            (request_lifecycle, serde_json::from_slice::<Value>(body))
        {
            lifecycle.observe_non_stream_json(&value);
        }
    }

    // Helper to process prefill response and extract body if needed for logprobs
    async fn process_prefill_response(
        &self,
        prefill_response: reqwest::Response,
        prefill_url: &str,
        return_logprob: bool,
    ) -> Result<(StatusCode, Option<Bytes>), Response> {
        let prefill_status = StatusCode::from_u16(prefill_response.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

        // Check if prefill succeeded
        if !prefill_status.is_success() {
            // Get error body from prefill
            let error_msg = prefill_response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown prefill error".to_string());

            error!(
                "Prefill server returned error status prefill_url={} status={} body={}",
                prefill_url, prefill_status, error_msg
            );

            // Map prefill_status to appropriate error function
            let error_response = match prefill_status {
                StatusCode::BAD_REQUEST => error::bad_request(
                    "prefill_bad_request",
                    format!("Prefill server error ({prefill_status}): {error_msg}"),
                ),
                StatusCode::NOT_FOUND => error::not_found(
                    "prefill_not_found",
                    format!("Prefill server error ({prefill_status}): {error_msg}"),
                ),
                StatusCode::INTERNAL_SERVER_ERROR => error::internal_error(
                    "prefill_internal_error",
                    format!("Prefill server error ({prefill_status}): {error_msg}"),
                ),
                StatusCode::SERVICE_UNAVAILABLE => error::service_unavailable(
                    "prefill_unavailable",
                    format!("Prefill server error ({prefill_status}): {error_msg}"),
                ),
                StatusCode::BAD_GATEWAY => error::bad_gateway(
                    "prefill_bad_gateway",
                    format!("Prefill server error ({prefill_status}): {error_msg}"),
                ),
                _ => error::internal_error(
                    "prefill_error",
                    format!("Prefill server error ({prefill_status}): {error_msg}"),
                ),
            };
            return Err(error_response);
        }

        // Read prefill body if needed for logprob merging
        let prefill_body = if return_logprob {
            match prefill_response.bytes().await {
                Ok(body) => Some(body),
                Err(e) => {
                    warn!("Failed to read prefill response body for logprobs: {}", e);
                    None
                }
            }
        } else {
            // For non-logprob requests, just consume the response without storing
            debug!("Consuming prefill response body (non-logprob request)");
            match prefill_response.bytes().await {
                Ok(_) => debug!("Prefill response consumed successfully"),
                Err(e) => warn!("Error consuming prefill response: {}", e),
            }
            None
        };

        Ok((prefill_status, prefill_body))
    }

    #[expect(
        clippy::unused_self,
        reason = "method on PDRouter for consistent API; may use self.api_key in future"
    )]
    fn build_post_with_headers(
        &self,
        client: &Client,
        worker: &dyn Worker,
        route: &'static str,
        json_request: &Value,
        headers: Option<&HeaderMap>,
        connection_close: bool,
    ) -> reqwest::RequestBuilder {
        let endpoint_url = worker.endpoint_url(route);
        let mut request = client.post(endpoint_url).json(json_request);
        if connection_close {
            request = request.header("Connection", "close");
        }
        if let Some(headers) = headers {
            for (name, value) in headers {
                if header_utils::should_forward_request_header(name.as_str()) {
                    if let Ok(val) = value.to_str() {
                        request = request.header(name, val);
                    }
                }
            }
        }
        request
    }

    // Helper to merge logprobs from prefill and decode responses
    // Optimized to avoid double cloning by taking ownership of decode array
    fn merge_logprobs_in_json(prefill_json: &Value, decode_json: &mut Value) -> bool {
        if let (Some(prefill_meta), Some(decode_meta)) = (
            prefill_json.get("meta_info"),
            decode_json.get_mut("meta_info"),
        ) {
            if let (Some(prefill_logprobs), Some(decode_logprobs)) = (
                prefill_meta.get("input_token_logprobs"),
                decode_meta.get_mut("input_token_logprobs"),
            ) {
                if let Some(prefill_arr) = prefill_logprobs.as_array() {
                    // Take ownership of decode array to avoid cloning it
                    let decode_arr = std::mem::take(decode_logprobs);
                    if let Value::Array(decode_vec) = decode_arr {
                        // Pre-allocate merged array with exact capacity
                        let mut merged = Vec::with_capacity(prefill_arr.len() + decode_vec.len());
                        merged.extend(prefill_arr.iter().cloned());
                        merged.extend(decode_vec);
                        decode_meta["input_token_logprobs"] = Value::Array(merged);
                        return true;
                    }
                }
            }
        }
        false
    }

    // Simple helper to merge logprobs in streaming responses
    // Optimized to reduce allocations in the merge path
    fn merge_streaming_logprobs(
        prefill_logprobs: Option<&Value>,
        decode_json: Option<&mut Value>,
        encoder: &mut SseEncoder,
    ) -> Result<Bytes, ()> {
        let decode_json = decode_json.ok_or(())?;

        // Merge prefill logprobs if available
        if let Some(p_logprobs) = prefill_logprobs {
            if let Some(meta) = decode_json.get_mut("meta_info") {
                if let Some(d_logprobs) = meta.get_mut("input_token_logprobs") {
                    if let Some(p_arr) = p_logprobs.as_array() {
                        // Take ownership of decode array to avoid cloning it
                        let decode_arr = std::mem::take(d_logprobs);
                        if let Value::Array(d_vec) = decode_arr {
                            // Pre-allocate merged array with exact capacity
                            let mut merged = Vec::with_capacity(p_arr.len() + d_vec.len());
                            merged.extend(p_arr.iter().cloned());
                            merged.extend(d_vec);
                            *d_logprobs = Value::Array(merged);
                        }
                    }
                }
            }
        }

        // Re-serialize via the shared encoder (reuses its buffer across chunks).
        encoder.encode_data(decode_json).map_err(|_| ())
    }
}

#[async_trait]
impl RouterTrait for PDRouter {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn health_generate(&self, _req: Request<Body>) -> Response {
        // Note: This endpoint actually causes the model to generate tokens, so we only test one pair

        // Select a random worker pair using the policy
        let (prefill, decode) = match self
            .select_pd_pair(None, UNKNOWN_MODEL_ID, None, None)
            .await
        {
            Ok(pair) => pair,
            Err(e) => {
                return error::service_unavailable(
                    "no_healthy_worker_pair",
                    format!("No healthy worker pair available: {e}"),
                );
            }
        };

        let prefill_url = format!("{}/health_generate", prefill.url());
        let (prefill_result, decode_result) = tokio::join!(
            self.client.get(&prefill_url).send(),
            self.client
                .get(format!("{}/health_generate", decode.url()))
                .send()
        );

        // Check results
        let mut errors = Vec::new();

        match prefill_result {
            Ok(res) if res.status().is_success() => {
                debug!(
                    "Health generate passed for prefill server: {}",
                    prefill.url()
                );
            }
            Ok(res) => {
                errors.push(format!(
                    "Prefill {} returned status {}",
                    prefill.url(),
                    res.status()
                ));
            }
            Err(e) => {
                errors.push(format!("Prefill {} error: {}", prefill.url(), e));
            }
        }

        match decode_result {
            Ok(res) if res.status().is_success() => {
                debug!("Health generate passed for decode server: {}", decode.url());
            }
            Ok(res) => {
                errors.push(format!(
                    "Decode {} returned status {}",
                    decode.url(),
                    res.status()
                ));
            }
            Err(e) => {
                errors.push(format!("Decode {} error: {}", decode.url(), e));
            }
        }

        if errors.is_empty() {
            (
                StatusCode::OK,
                format!(
                    "Health generate passed on selected pair: prefill={}, decode={}",
                    prefill.url(),
                    decode.url()
                ),
            )
                .into_response()
        } else {
            error::service_unavailable(
                "health_generate_failed",
                format!("Health generate failed: {errors:?}"),
            )
        }
    }

    async fn get_server_info(&self, _req: Request<Body>) -> Response {
        self.proxy_to_first_prefill_worker("get_server_info", None)
            .await
    }

    async fn get_model_info(&self, req: Request<Body>) -> Response {
        // Extract headers first to avoid Send issues
        let headers = header_utils::copy_request_headers(&req);

        // Proxy to first prefill worker
        self.proxy_to_first_prefill_worker("get_model_info", Some(headers))
            .await
    }

    async fn route_generate(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: &GenerateRequest,
        model_id: &str,
    ) -> Response {
        let is_stream = body.stream;
        let return_logprob = body.return_logprob.unwrap_or(false);

        let request_text = if self.policies_need_request_text() {
            body.text.as_deref().map(|s| s.to_string())
        } else {
            None
        };

        let batch_size = Self::get_generate_batch_size(body);

        let context = PDRequestContext {
            route: "/generate",
            batch_size,
            is_stream,
            return_logprob,
            request_text,
            model_id,
            headers: headers.cloned(),
        };

        self.execute_dual_dispatch(headers, tenant_meta, body, context)
            .await
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: &ChatCompletionRequest,
        model_id: &str,
    ) -> Response {
        let is_stream = body.stream;
        let return_logprob = body.logprobs;

        let request_text = if self.policies_need_request_text() {
            body.messages.first().and_then(|msg| match msg {
                ChatMessage::User { content, .. } => match content {
                    MessageContent::Text(text) => Some(text.clone()),
                    MessageContent::Parts(_) => None,
                },
                ChatMessage::Developer { content, .. } => match content {
                    MessageContent::Text(text) => Some(text.clone()),
                    MessageContent::Parts(_) => None,
                },
                ChatMessage::System { content, .. } => Some(content.to_simple_string()),
                _ => None,
            })
        } else {
            None
        };

        // Calculate batch size
        let batch_size = Self::get_chat_batch_size(body);

        let context = PDRequestContext {
            route: "/v1/chat/completions",
            batch_size,
            is_stream,
            return_logprob,
            request_text,
            model_id,
            headers: headers.cloned(),
        };

        self.execute_dual_dispatch(headers, tenant_meta, body, context)
            .await
    }

    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: &CompletionRequest,
        model_id: &str,
    ) -> Response {
        let is_stream = body.stream;
        let return_logprob = body.logprobs.is_some();

        let request_text = if self.policies_need_request_text() {
            match &body.prompt {
                StringOrArray::String(s) => Some(s.clone()),
                StringOrArray::Array(v) => v.first().map(|s| s.to_string()),
            }
        } else {
            None
        };

        // Calculate batch size
        let batch_size = Self::get_completion_batch_size(body);

        let context = PDRequestContext {
            route: "/v1/completions",
            batch_size,
            is_stream,
            return_logprob,
            request_text,
            model_id,
            headers: headers.cloned(),
        };

        self.execute_dual_dispatch(headers, tenant_meta, body, context)
            .await
    }

    async fn route_rerank(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: &RerankRequest,
        model_id: &str,
    ) -> Response {
        // Extract text for cache-aware routing
        let req_text = if self.policies_need_request_text() {
            Some(body.query.clone())
        } else {
            None
        };

        let context = PDRequestContext {
            route: "/v1/rerank",
            batch_size: None,
            is_stream: false,
            return_logprob: false,
            request_text: req_text,
            model_id,
            headers: headers.cloned(),
        };

        self.execute_dual_dispatch(headers, tenant_meta, body, context)
            .await
    }

    fn router_type(&self) -> &'static str {
        "pd"
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::{
        config::PolicyConfig,
        worker::{BasicWorkerBuilder, WorkerType},
    };

    fn create_test_pd_router() -> PDRouter {
        let worker_registry = Arc::new(WorkerRegistry::new());
        let policy_registry = Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));

        PDRouter {
            worker_registry,
            policy_registry,
            client: Client::new(),
            retry_config: RetryConfig::default(),
            api_key: Some("test_api_key".to_string()),
            pd_lifecycle_config: None,
        }
    }

    fn create_test_worker(url: String, worker_type: WorkerType, healthy: bool) -> Box<dyn Worker> {
        let worker = BasicWorkerBuilder::new(url)
            .worker_type(worker_type)
            .build();
        let status = if healthy {
            openai_protocol::worker::WorkerStatus::Ready
        } else {
            openai_protocol::worker::WorkerStatus::NotReady
        };
        worker.set_status(status);
        Box::new(worker)
    }

    fn create_labeled_worker(
        url: &str,
        worker_type: WorkerType,
        labels: &[(&str, &str)],
    ) -> Arc<dyn Worker> {
        let worker = BasicWorkerBuilder::new(url)
            .worker_type(worker_type)
            .labels(
                labels
                    .iter()
                    .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
                    .collect::<HashMap<_, _>>(),
            )
            .build();
        worker.set_status(openai_protocol::worker::WorkerStatus::Ready);
        Arc::new(worker)
    }

    #[test]
    fn test_done_event_detection() {
        // Production-incident payload: a delta whose arguments contained the
        // literal sentinel text; the old substring scan treated it as
        // terminal and silently killed the stream.
        let incident: &[u8] = b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"function\":{\"arguments\":\"// data: [DONE]\"}}]}}]}\n\n";
        let cases: &[(&[u8], bool, &str)] = &[
            (b"data: [DONE]\n\n", true, "standalone sentinel"),
            (
                b"data: {\"x\":1}\n\ndata: [DONE]\n\n",
                true,
                "sentinel after a data event",
            ),
            (b"data: [DONE]\r\n\r\n", true, "CRLF endings"),
            (incident, false, "sentinel text inside a JSON payload"),
            (
                b"data: [DONE]{\"x\":1}\n\n",
                false,
                "line continues with payload",
            ),
            (
                b"data: [DONE]",
                false,
                "possibly a split payload line: defer",
            ),
            (
                b"data: [DONE]\n",
                false,
                "event delimiter incomplete: defer",
            ),
            (
                b"data: [DONE]\ndata: x\n\n",
                false,
                "one event, joined data is not [DONE]",
            ),
        ];
        for (chunk, expected, case) in cases {
            let mut parser = PdSseParser::default();
            assert_eq!(
                parser.observe_chunk(chunk, |_| {}).done,
                *expected,
                "{case}"
            );
        }

        let mut split = PdSseParser::default();
        assert!(!split.observe_chunk(b"data: [DO", |_| {}).done);
        assert!(split.observe_chunk(b"NE]\n\n", |_| {}).done);

        let mut multiline = PdSseParser::default();
        assert!(
            !multiline
                .observe_chunk(b"data: [DONE]\ndata: x\n\n", |_| {})
                .done
        );
    }

    #[test]
    fn lifecycle_success_waits_for_downstream_terminal_marker() {
        use metrics_exporter_prometheus::PrometheusBuilder;

        use crate::observability::pd_request_lifecycle::TrafficClass;

        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            let config = Arc::new(
                PdLifecycleConfig::new(
                    "test",
                    "glm52",
                    "stable",
                    "smg-pd-request-v1",
                    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                    "0744ea180744ea180744ea180744ea180744ea18",
                )
                .expect("valid lifecycle config"),
            );
            let request = PdRequestLifecycle::start(config, TrafficClass::Natural);
            let (tx, rx) = mpsc::channel(4);
            tx.try_send(PdRelayEvent::Data {
                result: Ok(Bytes::from_static(
                    b"data: {\"text\":\"a\",\"meta_info\":{\"prompt_tokens\":8,\"completion_tokens\":1}}\n\n",
                )),
                observed_events: vec![serde_json::json!({
                    "text": "a",
                    "meta_info": {"prompt_tokens": 8, "completion_tokens": 1}
                })],
                observation_overflowed: false,
            })
            .expect("receiver is open");
            tx.try_send(PdRelayEvent::Terminal(PdTerminalOutcome::Success))
                .expect("receiver is open");
            drop(tx);

            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("test runtime");
            runtime.block_on(async {
                let mut stream = PdLifecycleRelayStream::new(rx, Some(request), StatusCode::OK);
                assert!(stream.next().await.is_some());
                assert!(
                    !handle.render().contains("smg_pd_terminal_requests_total{"),
                    "upstream relay completion must not finish the request before downstream polls the terminal marker"
                );
                assert!(stream.next().await.is_none());
            });
        });

        assert_eq!(
            handle
                .render()
                .matches("smg_pd_terminal_requests_total{")
                .count(),
            1
        );
    }

    #[test]
    fn sse_application_error_followed_by_done_is_terminal_error() {
        use metrics_exporter_prometheus::PrometheusBuilder;

        use crate::observability::pd_request_lifecycle::TrafficClass;

        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("test runtime");
            runtime.block_on(async {
                let router = create_test_pd_router();
                let config = Arc::new(
                    PdLifecycleConfig::new(
                        "test",
                        "glm52",
                        "stable",
                        "smg-pd-request-v1",
                        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                        "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                        "0744ea180744ea180744ea180744ea180744ea18",
                    )
                    .expect("valid lifecycle config"),
                );
                let request = PdRequestLifecycle::start(config, TrafficClass::Natural);
                let upstream = tokio_stream::iter(vec![
                    Ok::<Bytes, reqwest::Error>(Bytes::from_static(
                        b"data: {\"error\":{\"message\":\"partial failure\"}}\n\n",
                    )),
                    Ok::<Bytes, reqwest::Error>(Bytes::from_static(b"data: [DONE]\n\n")),
                ]);
                let response = router.create_streaming_response(
                    upstream,
                    StatusCode::OK,
                    None,
                    false,
                    None,
                    None,
                    Vec::new(),
                    Some(request),
                );
                axum::body::to_bytes(response.into_body(), usize::MAX)
                    .await
                    .expect("relay body");
            });
        });

        let rendered = handle.render();
        assert_eq!(
            rendered.matches("smg_pd_terminal_requests_total{").count(),
            1
        );
        assert!(rendered.lines().any(|line| {
            line.starts_with("smg_pd_terminal_requests_total{")
                && line.contains(r#"outcome="error""#)
        }));
        assert!(rendered.lines().any(|line| {
            line.starts_with("smg_pd_request_metric_evidence_total{")
                && line.contains(r#"metric="usage""#)
                && line.contains(r#"evidence_state="unknown""#)
        }));
    }

    #[test]
    fn oversized_sse_event_batch_is_forwarded_but_metrics_fail_closed() {
        use metrics_exporter_prometheus::PrometheusBuilder;

        use crate::observability::pd_request_lifecycle::TrafficClass;

        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let mut payload = String::new();
        for event_index in 0..129 {
            let event = if event_index == 128 {
                serde_json::json!({
                    "text": "x",
                    "meta_info": {"prompt_tokens": 8, "completion_tokens": 129}
                })
            } else {
                serde_json::json!({"text": "x"})
            };
            payload.push_str(&format!("data: {event}\n\n"));
        }
        payload.push_str("data: [DONE]\n\n");

        metrics::with_local_recorder(&recorder, || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("test runtime");
            runtime.block_on(async {
                let router = create_test_pd_router();
                let config = Arc::new(
                    PdLifecycleConfig::new(
                        "test",
                        "glm52",
                        "stable",
                        "smg-pd-request-v1",
                        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                        "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                        "0744ea180744ea180744ea180744ea180744ea18",
                    )
                    .expect("valid lifecycle config"),
                );
                let request = PdRequestLifecycle::start(config, TrafficClass::Natural);
                let upstream = tokio_stream::iter(vec![Ok::<Bytes, reqwest::Error>(Bytes::from(
                    payload.clone(),
                ))]);
                let response = router.create_streaming_response(
                    upstream,
                    StatusCode::OK,
                    None,
                    false,
                    None,
                    None,
                    Vec::new(),
                    Some(request),
                );
                let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                    .await
                    .expect("relay body");
                assert_eq!(body.as_ref(), payload.as_bytes());
            });
        });

        let rendered = handle.render();
        for metric in ["usage", "ttft", "tpot"] {
            assert!(rendered.lines().any(|line| {
                line.starts_with("smg_pd_request_metric_evidence_total{")
                    && line.contains(&format!(r#"metric="{metric}""#))
                    && line.contains(r#"evidence_state="unknown""#)
            }));
        }
        assert!(!rendered.contains("smg_pd_tpot_seconds_count{"));
    }

    #[test]
    fn oversized_single_sse_event_is_forwarded_but_metrics_fail_closed() {
        use metrics_exporter_prometheus::PrometheusBuilder;

        use crate::observability::pd_request_lifecycle::TrafficClass;

        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let oversized_token = "x".repeat(64 * 1024 + 1);
        let payload = format!(
            concat!(
                "data: {{\"text\":\"a\",\"meta_info\":{{\"completion_tokens\":1}}}}\n\n",
                "data: {{\"text\":\"b\",\"meta_info\":{{\"completion_tokens\":2}}}}\n\n",
                "data: {{\"text\":\"{}\",\"meta_info\":{{\"completion_tokens\":3}}}}\n\n",
                "data: {{\"meta_info\":{{\"prompt_tokens\":8,\"completion_tokens\":3}}}}\n\n",
                "data: [DONE]\n\n"
            ),
            oversized_token
        );

        metrics::with_local_recorder(&recorder, || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("test runtime");
            runtime.block_on(async {
                let router = create_test_pd_router();
                let config = Arc::new(
                    PdLifecycleConfig::new(
                        "test",
                        "glm52",
                        "stable",
                        "smg-pd-request-v1",
                        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                        "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                        "0744ea180744ea180744ea180744ea180744ea18",
                    )
                    .expect("valid lifecycle config"),
                );
                let request = PdRequestLifecycle::start(config, TrafficClass::Natural);
                let upstream = tokio_stream::iter(vec![Ok::<Bytes, reqwest::Error>(Bytes::from(
                    payload.clone(),
                ))]);
                let response = router.create_streaming_response(
                    upstream,
                    StatusCode::OK,
                    None,
                    false,
                    None,
                    None,
                    Vec::new(),
                    Some(request),
                );
                let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                    .await
                    .expect("relay body");
                assert_eq!(body.as_ref(), payload.as_bytes());
            });
        });

        let rendered = handle.render();
        for metric in ["usage", "ttft", "tpot"] {
            assert!(rendered.lines().any(|line| {
                line.starts_with("smg_pd_request_metric_evidence_total{")
                    && line.contains(&format!(r#"metric="{metric}""#))
                    && line.contains(r#"evidence_state="unknown""#)
            }));
        }
        assert!(!rendered.contains("smg_pd_completed_input_tokens_total{"));
        assert!(!rendered.contains("smg_pd_completed_output_tokens_total{"));
        assert!(!rendered.contains("smg_pd_request_v1_ttft_seconds_count{"));
        assert!(!rendered.contains("smg_pd_tpot_seconds_count{"));
    }

    #[test]
    fn oversized_sse_event_skips_logprob_merge_and_preserves_bytes() {
        use metrics_exporter_prometheus::PrometheusBuilder;

        use crate::observability::pd_request_lifecycle::TrafficClass;

        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let oversized_token = "x".repeat(64 * 1024 + 1);
        let first_chunk = format!(
            concat!(
                "data: {{\"text\":\"a\",\"meta_info\":{{",
                "\"input_token_logprobs\":[[-0.2,1,\"a\"]],",
                "\"completion_tokens\":1}}}}\n\n",
                "data: {{\"text\":\"{}\",\"meta_info\":",
                "{{\"completion_tokens\":2}}}}\n\n"
            ),
            oversized_token
        );
        let final_chunk = concat!(
            "data: {\"meta_info\":{\"prompt_tokens\":8,\"completion_tokens\":2}}\n\n",
            "data: [DONE]\n\n"
        );
        let expected = format!("{first_chunk}{final_chunk}");

        metrics::with_local_recorder(&recorder, || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("test runtime");
            runtime.block_on(async {
                let router = create_test_pd_router();
                let config = Arc::new(
                    PdLifecycleConfig::new(
                        "test",
                        "glm52",
                        "stable",
                        "smg-pd-request-v1",
                        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                        "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                        "0744ea180744ea180744ea180744ea180744ea18",
                    )
                    .expect("valid lifecycle config"),
                );
                let request = PdRequestLifecycle::start(config, TrafficClass::Natural);
                let upstream = tokio_stream::iter(vec![
                    Ok::<Bytes, reqwest::Error>(Bytes::from(first_chunk)),
                    Ok::<Bytes, reqwest::Error>(Bytes::from_static(final_chunk.as_bytes())),
                ]);
                let response = router.create_streaming_response(
                    upstream,
                    StatusCode::OK,
                    Some(serde_json::json!([[-0.1, 0, "prefill"]])),
                    true,
                    None,
                    None,
                    Vec::new(),
                    Some(request),
                );
                let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                    .await
                    .expect("relay body");
                assert_eq!(body.as_ref(), expected.as_bytes());
            });
        });

        let rendered = handle.render();
        for metric in ["usage", "ttft", "tpot"] {
            assert!(rendered.lines().any(|line| {
                line.starts_with("smg_pd_request_metric_evidence_total{")
                    && line.contains(&format!(r#"metric="{metric}""#))
                    && line.contains(r#"evidence_state="unknown""#)
            }));
        }
    }

    #[test]
    fn test_merge_streaming_logprobs_sentinel_exact_match() {
        let mut encoder = SseEncoder::new();
        // The exact sentinel produces no parsed JSON event and is forwarded.
        assert!(PDRouter::merge_streaming_logprobs(None, None, &mut encoder).is_err());
        // A payload containing "[DONE]" as text is still processed
        let mut event = serde_json::json!({"text": "[DONE]", "meta_info": {}});
        assert!(PDRouter::merge_streaming_logprobs(None, Some(&mut event), &mut encoder).is_ok());
    }

    #[test]
    fn pd_response_boundary_strips_internal_probe_headers() {
        let mut response = Response::new(Body::empty());
        for name in [
            "x-smg-pd-service",
            "x-smg-pd-release",
            "x-smg-pd-environment",
            "x-smg-pd-release-generation",
            "x-smg-pd-membership-checksum",
            "x-smg-pd-cache-domain",
            "x-smg-pd-traffic-class",
        ] {
            response
                .headers_mut()
                .insert(name, HeaderValue::from_static("forged"));
        }
        response
            .headers_mut()
            .insert("x-request-id", HeaderValue::from_static("kept"));

        PDRouter::finalize_pd_response(&mut response, None);

        for name in [
            "x-smg-pd-service",
            "x-smg-pd-release",
            "x-smg-pd-environment",
            "x-smg-pd-release-generation",
            "x-smg-pd-membership-checksum",
            "x-smg-pd-cache-domain",
            "x-smg-pd-traffic-class",
        ] {
            assert!(!response.headers().contains_key(name), "leaked {name}");
        }
        assert_eq!(response.headers()["x-request-id"], "kept");
    }

    #[test]
    fn test_build_post_uses_dp_base_url_for_logical_worker() {
        let router = create_test_pd_router();
        let worker = BasicWorkerBuilder::new("http://127.0.0.1:30000")
            .worker_type(WorkerType::Decode)
            .dp_config(2, 4)
            .build();

        let request = router
            .build_post_with_headers(
                &router.client,
                &worker,
                "/generate",
                &json!({"text": "hello"}),
                None,
                false,
            )
            .build()
            .expect("request should build");

        assert_eq!(worker.url(), "http://127.0.0.1:30000@2");
        assert_eq!(
            worker.endpoint_url("/generate"),
            "http://127.0.0.1:30000/generate"
        );
        assert_eq!(request.url().as_str(), "http://127.0.0.1:30000/generate");
    }

    #[tokio::test]
    async fn test_select_healthy_prefill_worker() {
        let router = create_test_pd_router();

        let healthy_worker =
            create_test_worker("http://healthy".to_string(), WorkerType::Prefill, true);
        let unhealthy_worker =
            create_test_worker("http://unhealthy".to_string(), WorkerType::Prefill, false);
        let decode_worker =
            create_test_worker("http://decode".to_string(), WorkerType::Decode, true);

        router
            .worker_registry
            .register_or_replace(Arc::from(unhealthy_worker));
        router
            .worker_registry
            .register_or_replace(Arc::from(healthy_worker));
        router
            .worker_registry
            .register_or_replace(Arc::from(decode_worker));

        let result = router
            .select_pd_pair(None, UNKNOWN_MODEL_ID, None, None)
            .await;

        assert!(result.is_ok());
        let (prefill, _decode) = result.unwrap();

        assert_eq!(prefill.url(), "http://healthy");
        assert!(prefill.is_healthy());
    }

    #[tokio::test]
    async fn trusted_probe_cache_domain_filters_both_pd_legs() {
        let router = create_test_pd_router();
        for worker in [
            create_labeled_worker(
                "http://prefill-a",
                WorkerType::Prefill,
                &[
                    ("cache_domain", "cache-a"),
                    ("execution_cohort_pair_id", "pair-a"),
                ],
            ),
            create_labeled_worker(
                "http://decode-a",
                WorkerType::Decode,
                &[("execution_cohort_pair_id", "pair-a")],
            ),
            create_labeled_worker(
                "http://prefill-b",
                WorkerType::Prefill,
                &[
                    ("cache_domain", "cache-b"),
                    ("execution_cohort_pair_id", "pair-b"),
                ],
            ),
            create_labeled_worker(
                "http://decode-b",
                WorkerType::Decode,
                &[("execution_cohort_pair_id", "pair-b")],
            ),
        ] {
            router.worker_registry.register_or_replace(worker);
        }

        let (prefill, decode) = router
            .select_pd_pair(None, UNKNOWN_MODEL_ID, None, Some("cache-b"))
            .await
            .expect("catalogued cache-domain pair");
        assert_eq!(prefill.url(), "http://prefill-b");
        assert_eq!(decode.url(), "http://decode-b");
    }

    #[tokio::test]
    async fn test_empty_worker_lists() {
        let router = create_test_pd_router();

        let result = router
            .select_pd_pair(None, UNKNOWN_MODEL_ID, None, None)
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No prefill workers available"));
    }

    #[test]
    fn test_worker_load_metrics() {
        let prefill_worker: Arc<dyn Worker> = Arc::from(create_test_worker(
            "http://prefill".to_string(),
            WorkerType::Prefill,
            true,
        ));
        let decode_worker: Arc<dyn Worker> = Arc::from(create_test_worker(
            "http://decode".to_string(),
            WorkerType::Decode,
            true,
        ));

        let _prefill_guard = WorkerLoadGuard::new(prefill_worker.clone(), None);
        let _decode_guard = WorkerLoadGuard::new(decode_worker.clone(), None);

        assert_eq!(prefill_worker.load(), 1);
        assert_eq!(decode_worker.load(), 1);

        drop(_prefill_guard);
        drop(_decode_guard);

        assert_eq!(prefill_worker.load(), 0);
        assert_eq!(decode_worker.load(), 0);
    }

    #[tokio::test]
    async fn test_streaming_decode_error_emits_valid_json_sse() {
        let router = create_test_pd_router();

        let prefill: Arc<dyn Worker> = Arc::from(create_test_worker(
            "http://prefill".to_string(),
            WorkerType::Prefill,
            true,
        ));
        let decode: Arc<dyn Worker> = Arc::from(create_test_worker(
            "http://decode".to_string(),
            WorkerType::Decode,
            true,
        ));

        let upstream = http::Response::builder()
            .status(StatusCode::INTERNAL_SERVER_ERROR)
            .body(r#"{"error":"boom \"quoted\""}"#)
            .unwrap();
        let decode_response = reqwest::Response::from(upstream);

        let context = PDRequestContext {
            route: "/v1/chat/completions",
            batch_size: None,
            is_stream: true,
            return_logprob: false,
            request_text: None,
            model_id: UNKNOWN_MODEL_ID,
            headers: None,
        };

        let load_guards = vec![
            WorkerLoadGuard::new(prefill.clone(), None),
            WorkerLoadGuard::new(decode.clone(), None),
        ];

        let response = router
            .handle_decode_error_response(decode_response, &context, decode, load_guards)
            .await;

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let frame = std::str::from_utf8(&body).unwrap();

        let payload = frame
            .strip_prefix("data: ")
            .expect("SSE frame must start with `data: `")
            .trim_end();
        let parsed: Value =
            serde_json::from_str(payload).expect("bytes after `data: ` must be valid JSON");
        assert!(
            parsed.get("error").is_some(),
            "parsed SSE payload must contain an `error` field: {parsed}"
        );
    }

    #[tokio::test]
    async fn test_streaming_load_tracking() {
        use futures_util::StreamExt;
        use tokio::time::{sleep, Duration};
        use tokio_stream::wrappers::UnboundedReceiverStream;

        let router = create_test_pd_router();

        let prefill_worker =
            create_test_worker("http://prefill".to_string(), WorkerType::Prefill, true);
        let decode_worker =
            create_test_worker("http://decode".to_string(), WorkerType::Decode, true);

        router
            .worker_registry
            .register_or_replace(Arc::from(prefill_worker));
        router
            .worker_registry
            .register_or_replace(Arc::from(decode_worker));

        let prefill_workers = router.worker_registry.get_prefill_workers();
        let decode_workers = router.worker_registry.get_decode_workers();

        let prefill_ref = prefill_workers[0].clone();
        let decode_ref = decode_workers[0].clone();

        assert_eq!(prefill_ref.load(), 0);
        assert_eq!(decode_ref.load(), 0);

        let (tx, rx) = mpsc::unbounded_channel();
        let stream = UnboundedReceiverStream::new(rx);

        {
            let guards = vec![
                WorkerLoadGuard::new(prefill_ref.clone(), None),
                WorkerLoadGuard::new(decode_ref.clone(), None),
            ];

            assert_eq!(prefill_ref.load(), 1);
            assert_eq!(decode_ref.load(), 1);

            let response = router.create_streaming_response(
                stream.map(Ok),
                StatusCode::OK,
                None,
                false,
                None,
                None,
                guards,
                None,
            );

            // Guards are now attached to response body, so load should be 1
            assert_eq!(prefill_ref.load(), 1);
            assert_eq!(decode_ref.load(), 1);

            tx.send(Bytes::from("test data")).unwrap();

            sleep(Duration::from_millis(10)).await;

            // Load still 1 while response body exists
            assert_eq!(prefill_ref.load(), 1);
            assert_eq!(decode_ref.load(), 1);

            drop(tx);

            // Response (and its body with guards) dropped here
            drop(response);
        }

        // Guards dropped when response dropped
        assert_eq!(prefill_ref.load(), 0);
        assert_eq!(decode_ref.load(), 0);
    }
}
