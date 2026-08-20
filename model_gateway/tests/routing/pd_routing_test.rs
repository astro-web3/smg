//! Prefill/Decode (PD) routing integration tests
//!
//! Tests for prefill-decode disaggregation routing mode.

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use serde_json::json;
use smg::config::RouterConfig;
use tower::ServiceExt;

use crate::common::{
    mock_worker::{HealthStatus, MockWorkerConfig, WorkerType},
    AppTestContext, TestWorkerConfig,
};

#[cfg(test)]
mod pd_routing_tests {
    use super::*;

    /// Test basic PD mode routing with prefill and decode workers
    #[tokio::test]
    async fn test_pd_mode_basic_routing() {
        let mut config = RouterConfig::builder()
            .prefill_decode_mode(
                vec![
                    ("http://127.0.0.1:19800".to_string(), None),
                    ("http://127.0.0.1:19801".to_string(), None),
                ],
                vec![
                    "http://127.0.0.1:19802".to_string(),
                    "http://127.0.0.1:19803".to_string(),
                ],
            )
            .power_of_two_policy(1)
            .host("127.0.0.1")
            .port(3800)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .build_unchecked();
        config.health_check.disable_health_check = true;

        // Note: For PD mode tests, we need to start prefill and decode workers separately
        // The test context will need to handle this specially
        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                // Prefill workers
                TestWorkerConfig::prefill(19800),
                TestWorkerConfig::prefill(19801),
                // Decode workers
                TestWorkerConfig::decode(19802),
                TestWorkerConfig::decode(19803),
            ],
        )
        .await;

        let app = ctx.create_app();

        // Send requests and verify they succeed
        for i in 0..10 {
            let payload = json!({
                "text": format!("PD mode request {}", i),
                "stream": false
            });

            let req = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap();

            let resp = app.clone().oneshot(req).await.unwrap();
            assert_eq!(
                resp.status(),
                StatusCode::OK,
                "PD mode request should succeed"
            );
        }

        ctx.shutdown().await;
    }

    /// Test PD mode with round robin policy
    #[tokio::test]
    async fn test_pd_mode_round_robin() {
        let mut config = RouterConfig::builder()
            .prefill_decode_mode(
                vec![("http://127.0.0.1:19810".to_string(), None)],
                vec![
                    "http://127.0.0.1:19811".to_string(),
                    "http://127.0.0.1:19812".to_string(),
                ],
            )
            .round_robin_policy()
            .host("127.0.0.1")
            .port(3801)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .build_unchecked();
        config.health_check.disable_health_check = true;

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::prefill(19810),
                TestWorkerConfig::decode(19811),
                TestWorkerConfig::decode(19812),
            ],
        )
        .await;

        let app = ctx.create_app();
        let mut success_count = 0;

        for i in 0..20 {
            let payload = json!({
                "text": format!("PD round robin {}", i),
                "stream": false
            });

            let req = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap();

            let resp = app.clone().oneshot(req).await.unwrap();
            if resp.status() == StatusCode::OK {
                success_count += 1;
            }
        }

        assert_eq!(
            success_count, 20,
            "All requests should succeed in PD mode with round robin"
        );

        ctx.shutdown().await;
    }

    /// A non-streaming PD request must emit the SMG-only PD metrics, including
    /// the pre-existing transport-level `smg_pd_ttft_seconds`. Runs on a current-thread runtime so the
    /// thread-local Prometheus recorder captures emissions from the request path.
    #[test]
    #[serial_test::serial]
    fn test_pd_metrics_emitted_on_request() {
        use metrics_exporter_prometheus::PrometheusBuilder;

        const PD_METRICS_ENV: [(&str, &str); 7] = [
            ("SMG_PD_ENVIRONMENT", "test"),
            ("SMG_PD_SERVICE", "glm52"),
            ("SMG_PD_RELEASE", "stable"),
            ("SMG_PD_METRIC_CONTRACT_VERSION", "smg-pd-request-v1"),
            (
                "SMG_PD_SCHEMA_DIGEST",
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            ),
            (
                "SMG_PD_IMAGE_DIGEST",
                "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            ),
            (
                "SMG_PD_PRODUCER_REVISION",
                "0744ea180744ea180744ea180744ea180744ea18",
            ),
        ];

        struct PdMetricsEnvGuard;
        impl PdMetricsEnvGuard {
            fn install() -> Self {
                for (key, value) in PD_METRICS_ENV {
                    std::env::set_var(key, value);
                }
                Self
            }
        }
        impl Drop for PdMetricsEnvGuard {
            fn drop(&mut self) {
                for (key, _) in PD_METRICS_ENV {
                    std::env::remove_var(key);
                }
            }
        }

        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let _env = PdMetricsEnvGuard::install();

        metrics::with_local_recorder(&recorder, || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(async {
                let mut config = RouterConfig::builder()
                    .prefill_decode_mode(
                        vec![("http://127.0.0.1:19830".to_string(), None)],
                        vec!["http://127.0.0.1:19831".to_string()],
                    )
                    .round_robin_policy()
                    .host("127.0.0.1")
                    .port(3803)
                    .max_payload_size(256 * 1024 * 1024)
                    .request_timeout_secs(600)
                    .worker_startup_timeout_secs(5)
                    .worker_startup_check_interval_secs(1)
                    .max_concurrent_requests(64)
                    .queue_timeout_secs(60)
                    .build_unchecked();
                config.health_check.disable_health_check = true;

                let ctx = AppTestContext::new_with_config(
                    config,
                    vec![
                        TestWorkerConfig::prefill(19830),
                        TestWorkerConfig::decode(19831),
                    ],
                )
                .await;

                let app = ctx.create_app();
                let payload = json!({ "text": "PD metrics request", "stream": false });
                let req = Request::builder()
                    .method("POST")
                    .uri("/generate")
                    .header(CONTENT_TYPE, "application/json")
                    .header("x-smg-traffic-class", "synthetic")
                    .body(Body::from(serde_json::to_string(&payload).unwrap()))
                    .unwrap();

                let resp = app.oneshot(req).await.unwrap();
                assert_eq!(resp.status(), StatusCode::OK, "PD request should succeed");

                ctx.shutdown().await;
            });
        });

        let rendered = handle.render();
        assert!(
            rendered.contains("smg_pd_prefill_duration_seconds_count"),
            "smg_pd_prefill_duration_seconds not emitted; rendered:\n{rendered}"
        );
        assert!(
            rendered.contains("smg_pd_ttft_seconds_count"),
            "smg_pd_ttft_seconds not emitted; rendered:\n{rendered}"
        );
        assert!(
            rendered.contains("smg_pd_request_lifecycle_contract_info{")
                && rendered.contains(r#"metric_contract_version="smg-pd-request-v1""#),
            "PD lifecycle capability anchor not emitted; rendered:\n{rendered}"
        );
        assert_eq!(
            rendered.matches("smg_pd_requests_started_total{").count(),
            1,
            "request start must be emitted once; rendered:\n{rendered}"
        );
        assert_eq!(
            rendered.matches("smg_pd_terminal_requests_total{").count(),
            1,
            "request terminal must be emitted once; rendered:\n{rendered}"
        );
        assert!(
            rendered.contains(r#"traffic_class="natural""#)
                && !rendered.contains(r#"traffic_class="synthetic""#),
            "public client traffic class must fail safe to natural; rendered:\n{rendered}"
        );
        assert!(
            rendered.contains("smg_pd_completed_input_tokens_total{")
                && rendered.contains("smg_pd_completed_output_tokens_total{"),
            "validated terminal usage counters not emitted; rendered:\n{rendered}"
        );
        assert!(
            rendered.contains(r#"execution_cohort_pair_id="unassigned""#),
            "workers without trusted cohort metadata must fail closed; rendered:\n{rendered}"
        );
        assert!(
            rendered.contains(r#"workload_bucket="unassigned""#),
            "public traffic without a trusted request classifier must not inherit an operator-wide workload bucket; rendered:\n{rendered}"
        );
    }

    /// Test PD mode handles worker failures gracefully
    #[tokio::test]
    async fn test_pd_mode_with_failing_decode_worker() {
        use smg::config::RetryConfig;

        let mut config = RouterConfig::builder()
            .prefill_decode_mode(
                vec![("http://127.0.0.1:19820".to_string(), None)],
                vec![
                    "http://127.0.0.1:19821".to_string(),
                    "http://127.0.0.1:19822".to_string(),
                ],
            )
            .round_robin_policy()
            .host("127.0.0.1")
            .port(3802)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .retry_config(RetryConfig {
                max_retries: 3,
                initial_backoff_ms: 10,
                max_backoff_ms: 50,
                ..Default::default()
            })
            .build_unchecked();
        config.health_check.disable_health_check = true;

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::prefill(19820),
                MockWorkerConfig {
                    port: 19821,
                    worker_type: WorkerType::Decode,
                    health_status: HealthStatus::Healthy,
                    response_delay_ms: 0,
                    fail_rate: 1.0, // Failing decode worker
                },
                TestWorkerConfig::decode(19822), // Healthy decode worker
            ],
        )
        .await;

        let app = ctx.create_app();

        // Request should succeed via retry to healthy decode worker
        let payload = json!({
            "text": "Test with failing decode worker",
            "stream": false
        });

        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "Request should succeed via retry to healthy decode worker"
        );

        ctx.shutdown().await;
    }
}
