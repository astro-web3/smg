//! Service-level request lifecycle metrics for HTTP PD routing.
//!
//! The request object is created outside the retry loop. This makes request
//! starts and final outcomes exactly-once even when worker attempts are
//! retried. Runtime labels come only from operator configuration and selected
//! worker metadata; no client-provided value is used as a metric label.

use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};

use axum::http::{header::InvalidHeaderValue, HeaderMap, HeaderValue, StatusCode};
use metrics::{counter, gauge, histogram};
use serde::Deserialize;
use serde_json::Value;

use super::metrics::Metrics;
use crate::{middleware::TenantRequestMeta, worker::Worker};

const ENV_PD_ENVIRONMENT: &str = "SMG_PD_ENVIRONMENT";
const ENV_PD_SERVICE: &str = "SMG_PD_SERVICE";
const ENV_PD_RELEASE: &str = "SMG_PD_RELEASE";
const ENV_CONTRACT_VERSION: &str = "SMG_PD_METRIC_CONTRACT_VERSION";
const ENV_SCHEMA_DIGEST: &str = "SMG_PD_SCHEMA_DIGEST";
const ENV_IMAGE_DIGEST: &str = "SMG_PD_IMAGE_DIGEST";
const ENV_PRODUCER_REVISION: &str = "SMG_PD_PRODUCER_REVISION";
const ENV_REQUEST_CLASS_CATALOG: &str = "SMG_PD_REQUEST_CLASS_CATALOG_JSON";
const ENV_COHORT_CATALOG: &str = "SMG_PD_COHORT_CATALOG_JSON";
const REQUEST_CLASS_CATALOG_VERSION: &str = "request-class-v1";
const COHORT_CATALOG_VERSION: &str = "cohort-pair-v1";
const MAX_NATURAL_PRINCIPALS: usize = 256;
const MAX_PRIVILEGED_ROUTES: usize = 16;
const MAX_WORKLOAD_BUCKETS: usize = 64;
const MAX_COHORT_PAIRS: usize = 256;
const MAX_CACHE_DOMAINS_PER_ROUTE: usize = 16;
const MAX_CATALOG_CACHE_DOMAINS: usize = 64;
const PROBE_CACHE_DOMAIN_HEADER: &str = "x-pd-probe-cache-domain";
const PROBE_RESPONSE_SERVICE_HEADER: &str = "x-smg-pd-service";
const PROBE_RESPONSE_RELEASE_HEADER: &str = "x-smg-pd-release";
const PROBE_RESPONSE_ENVIRONMENT_HEADER: &str = "x-smg-pd-environment";
const PROBE_RESPONSE_RELEASE_GENERATION_HEADER: &str = "x-smg-pd-release-generation";
const PROBE_RESPONSE_MEMBERSHIP_CHECKSUM_HEADER: &str = "x-smg-pd-membership-checksum";
const PROBE_RESPONSE_CACHE_DOMAIN_HEADER: &str = "x-smg-pd-cache-domain";
const PROBE_RESPONSE_TRAFFIC_CLASS_HEADER: &str = "x-smg-pd-traffic-class";
const PROBE_RESPONSE_HEADERS: [&str; 7] = [
    PROBE_RESPONSE_SERVICE_HEADER,
    PROBE_RESPONSE_RELEASE_HEADER,
    PROBE_RESPONSE_ENVIRONMENT_HEADER,
    PROBE_RESPONSE_RELEASE_GENERATION_HEADER,
    PROBE_RESPONSE_MEMBERSHIP_CHECKSUM_HEADER,
    PROBE_RESPONSE_CACHE_DOMAIN_HEADER,
    PROBE_RESPONSE_TRAFFIC_CLASS_HEADER,
];

const WORKER_RUNTIME_COHORT_LABEL: &str = "runtime_cohort";
const WORKER_CACHE_DOMAIN_LABEL: &str = "cache_domain";
const WORKER_COHORT_PAIR_LABEL: &str = "execution_cohort_pair_id";
const UNASSIGNED: &str = "unassigned";
const MAX_IDENTITY_LEN: usize = 63;
const MAX_SSE_OBSERVER_BUFFER: usize = 64 * 1024;
// Timing is only defined for one output choice. Retaining more identities has
// no diagnostic value after evidence has failed closed.
const MAX_TRACKED_CHOICE_INDICES: usize = 1;

/// Validated, operator-controlled identity for the PD lifecycle contract.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PdLifecycleConfig {
    environment: Arc<str>,
    pd_service: Arc<str>,
    smg_release: Arc<str>,
    metric_contract_version: Arc<str>,
    schema_digest: Arc<str>,
    image_digest: Arc<str>,
    producer_revision: Arc<str>,
    request_class_catalog: Option<Arc<RequestClassCatalog>>,
    cohort_catalog: Option<Arc<CohortCatalog>>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct RequestClassCatalogDocument {
    version: String,
    #[serde(default)]
    natural_principals: BTreeMap<String, String>,
    #[serde(default)]
    privileged_routes: BTreeMap<String, PrivilegedRoutePolicyDocument>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct PrivilegedRoutePolicyDocument {
    authenticated_principal: String,
    endpoint: String,
    traffic_class: String,
    workload_bucket: String,
    pd_service: String,
    smg_release: String,
    release_generation: String,
    membership_checksum: String,
    cache_domains: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RequestClassCatalog {
    version: Arc<str>,
    natural_principals: BTreeMap<String, Arc<str>>,
    privileged_routes: BTreeMap<String, PrivilegedRoutePolicy>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PrivilegedRoutePolicy {
    authenticated_principal: Arc<str>,
    endpoint: Arc<str>,
    traffic_class: TrafficClass,
    workload_bucket: Arc<str>,
    cache_domains: BTreeSet<Arc<str>>,
    response_attribution: PdResponseAttribution,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct CohortCatalogDocument {
    version: String,
    pairs: BTreeMap<String, CohortPairPolicy>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct CohortPairPolicy {
    cache_domain: String,
    prefill_runtime_cohort: String,
    decode_runtime_cohort: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CohortCatalog {
    version: Arc<str>,
    pairs: BTreeMap<String, CohortPairPolicy>,
}

impl PdLifecycleConfig {
    /// Build and validate a PD request-lifecycle identity.
    pub fn new(
        environment: &str,
        pd_service: &str,
        smg_release: &str,
        metric_contract_version: &str,
        schema_digest: &str,
        image_digest: &str,
        producer_revision: &str,
    ) -> Result<Self, String> {
        Self::new_with_catalogs(
            environment,
            pd_service,
            smg_release,
            metric_contract_version,
            schema_digest,
            image_digest,
            producer_revision,
            None,
            None,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the seven contract identity fields and two optional catalogs are validated atomically"
    )]
    fn new_with_catalogs(
        environment: &str,
        pd_service: &str,
        smg_release: &str,
        metric_contract_version: &str,
        schema_digest: &str,
        image_digest: &str,
        producer_revision: &str,
        request_class_catalog: Option<RequestClassCatalog>,
        cohort_catalog: Option<CohortCatalog>,
    ) -> Result<Self, String> {
        validate_environment(environment)?;
        validate_identity("pd_service", pd_service)?;
        validate_identity("smg_release", smg_release)?;
        validate_identity("metric_contract_version", metric_contract_version)?;
        if metric_contract_version != "smg-pd-request-v1" {
            return Err("this producer only supports smg-pd-request-v1".to_string());
        }
        validate_digest("schema_digest", schema_digest, false)?;
        validate_digest("image_digest", image_digest, true)?;
        validate_revision(producer_revision)?;

        Ok(Self {
            environment: Arc::from(environment),
            pd_service: Arc::from(pd_service),
            smg_release: Arc::from(smg_release),
            metric_contract_version: Arc::from(metric_contract_version),
            schema_digest: Arc::from(schema_digest),
            image_digest: Arc::from(image_digest),
            producer_revision: Arc::from(producer_revision),
            request_class_catalog: request_class_catalog.map(Arc::new),
            cohort_catalog: cohort_catalog.map(Arc::new),
        })
    }

    /// Load the optional contract from environment variables.
    ///
    /// No variables means disabled for backwards compatibility. A partial or
    /// malformed configuration is rejected rather than silently emitting an
    /// incomplete capability anchor.
    pub fn from_env() -> Result<Option<Self>, String> {
        Self::from_lookup(|key| env::var(key).ok())
    }

    pub(crate) fn from_lookup(
        mut lookup: impl FnMut(&str) -> Option<String>,
    ) -> Result<Option<Self>, String> {
        let keys = [
            ENV_PD_ENVIRONMENT,
            ENV_PD_SERVICE,
            ENV_PD_RELEASE,
            ENV_CONTRACT_VERSION,
            ENV_SCHEMA_DIGEST,
            ENV_IMAGE_DIGEST,
            ENV_PRODUCER_REVISION,
        ];
        let values: Vec<Option<String>> = keys.iter().map(|key| lookup(key)).collect();
        let request_class_catalog_json = lookup(ENV_REQUEST_CLASS_CATALOG);
        let cohort_catalog_json = lookup(ENV_COHORT_CATALOG);

        if values.iter().all(Option::is_none) {
            if request_class_catalog_json.is_some() || cohort_catalog_json.is_some() {
                return Err(
                    "PD catalogs cannot be configured while lifecycle metrics are disabled"
                        .to_string(),
                );
            }
            return Ok(None);
        }

        let required = |index: usize| {
            values[index]
                .as_deref()
                .filter(|value| !value.trim().is_empty())
                .ok_or_else(|| {
                    format!(
                        "{} is required when PD lifecycle metrics are enabled",
                        keys[index]
                    )
                })
        };

        let request_class_catalog = request_class_catalog_json
            .map(|json| parse_request_class_catalog(&json))
            .transpose()?;
        let cohort_catalog = cohort_catalog_json
            .map(|json| parse_cohort_catalog(&json))
            .transpose()?;

        Self::new_with_catalogs(
            required(0)?,
            required(1)?,
            required(2)?,
            required(3)?,
            required(4)?,
            required(5)?,
            required(6)?,
            request_class_catalog,
            cohort_catalog,
        )
        .map(Some)
    }

    /// Emit the zero-traffic capability anchor.
    pub fn install_capability_anchor(&self) {
        Metrics::install_pd_lifecycle_capability(self);
    }

    /// Resolve execution identity from trusted selected-worker metadata.
    /// Missing or invalid metadata remains explicitly unassigned.
    pub fn execution_identity(
        &self,
        prefill: &dyn Worker,
        decode: &dyn Worker,
    ) -> PdExecutionIdentity {
        self.execution_identity_from_labels(
            &prefill.metadata().spec.labels,
            &decode.metadata().spec.labels,
        )
    }

    fn execution_identity_from_labels(
        &self,
        prefill_labels: &std::collections::HashMap<String, String>,
        decode_labels: &std::collections::HashMap<String, String>,
    ) -> PdExecutionIdentity {
        let Some(cache_domain) = valid_worker_label(prefill_labels, WORKER_CACHE_DOMAIN_LABEL)
        else {
            return PdExecutionIdentity::unassigned();
        };
        let Some(prefill_cohort) = valid_worker_label(prefill_labels, WORKER_RUNTIME_COHORT_LABEL)
        else {
            return PdExecutionIdentity::unassigned();
        };
        let Some(decode_cohort) = valid_worker_label(decode_labels, WORKER_RUNTIME_COHORT_LABEL)
        else {
            return PdExecutionIdentity::unassigned();
        };
        let (Some(prefill_pair), Some(decode_pair)) = (
            valid_worker_label(prefill_labels, WORKER_COHORT_PAIR_LABEL),
            valid_worker_label(decode_labels, WORKER_COHORT_PAIR_LABEL),
        ) else {
            return PdExecutionIdentity::unassigned();
        };
        if prefill_pair != decode_pair {
            return PdExecutionIdentity::unassigned();
        }

        let Some(catalog) = self.cohort_catalog.as_ref() else {
            return PdExecutionIdentity::unassigned();
        };
        let Some(pair) = catalog.pairs.get(prefill_pair) else {
            return PdExecutionIdentity::unassigned();
        };
        if pair.cache_domain != cache_domain
            || pair.prefill_runtime_cohort != prefill_cohort
            || pair.decode_runtime_cohort != decode_cohort
        {
            return PdExecutionIdentity::unassigned();
        }

        PdExecutionIdentity {
            // Worker selection does not own request workload classification.
            // `PdRequestLifecycle::begin_attempt` overlays the bucket resolved
            // from authenticated server-side request metadata.
            workload_bucket: Arc::from(UNASSIGNED),
            cache_domain: Arc::from(cache_domain),
            prefill_runtime_cohort: Arc::from(prefill_cohort),
            decode_runtime_cohort: Arc::from(decode_cohort),
            execution_cohort_pair_id: Arc::from(prefill_pair),
        }
    }

    /// Classify HTTP traffic from server-owned metadata. Client headers are
    /// intentionally ignored. Privileged traffic additionally requires a
    /// trusted route extension, exact endpoint, and authenticated principal.
    pub fn classify_public_http_request(
        &self,
        request_meta: &TenantRequestMeta,
        endpoint: &str,
        headers: Option<&HeaderMap>,
    ) -> Result<PdRequestClassification, String> {
        let principal = request_meta.tenant_key().as_str();
        if let Some(route) = request_meta.extension::<PdTrustedRouteIdentity>() {
            let catalog = self.request_class_catalog.as_ref().ok_or_else(|| {
                "trusted PD route metadata is present but no request-class catalog is configured"
                    .to_string()
            })?;
            let policy = catalog
                .privileged_routes
                .get(route.as_str())
                .ok_or_else(|| {
                    "trusted PD route is not present in the request-class catalog".to_string()
                })?;
            if policy.authenticated_principal.as_ref() != principal
                || policy.endpoint.as_ref() != endpoint
            {
                return Err(
                    "trusted PD route, authenticated principal, and endpoint do not match"
                        .to_string(),
                );
            }
            if policy.response_attribution.pd_service != self.pd_service
                || policy.response_attribution.smg_release != self.smg_release
            {
                return Err(
                    "privileged route attribution does not match the active PD service/release"
                        .to_string(),
                );
            }
            let requested_cache_domain = trusted_probe_cache_domain(headers)?;
            let cache_domain = policy
                .cache_domains
                .get(requested_cache_domain)
                .ok_or_else(|| {
                    "requested probe cache domain is not present in the trusted route catalog"
                        .to_string()
                })?;
            let mut response_attribution = policy.response_attribution.clone();
            response_attribution.environment = Arc::clone(&self.environment);
            response_attribution.cache_domain = Arc::clone(cache_domain);
            return Ok(PdRequestClassification {
                traffic_class: policy.traffic_class,
                workload_bucket: Arc::clone(&policy.workload_bucket),
                response_attribution: Some(response_attribution),
            });
        }

        let workload_bucket = self
            .request_class_catalog
            .as_ref()
            .and_then(|catalog| catalog.natural_principals.get(principal))
            .cloned()
            .unwrap_or_else(|| Arc::from(UNASSIGNED));
        Ok(PdRequestClassification {
            traffic_class: TrafficClass::Natural,
            workload_bucket,
            response_attribution: None,
        })
    }

    /// Validate an opt-in internal HTTP route against the versioned catalog
    /// before the server attaches an unforgeable route marker to request meta.
    pub(crate) fn authorize_trusted_http_route(
        &self,
        request_meta: &TenantRequestMeta,
        route_identity: &str,
        canonical_endpoint: &str,
        headers: &HeaderMap,
    ) -> Result<PdTrustedRouteIdentity, String> {
        let identity = PdTrustedRouteIdentity::new(route_identity)?;
        let marked = request_meta.clone().with_extension(identity.clone());
        self.classify_public_http_request(&marked, canonical_endpoint, Some(headers))?;
        Ok(identity)
    }
}

fn parse_request_class_catalog(json: &str) -> Result<RequestClassCatalog, String> {
    let document: RequestClassCatalogDocument = serde_json::from_str(json)
        .map_err(|error| format!("invalid request-class catalog JSON: {error}"))?;
    if document.version != REQUEST_CLASS_CATALOG_VERSION {
        return Err(format!(
            "request-class catalog version must be {REQUEST_CLASS_CATALOG_VERSION}"
        ));
    }
    if document.natural_principals.len() > MAX_NATURAL_PRINCIPALS {
        return Err(format!(
            "request-class catalog supports at most {MAX_NATURAL_PRINCIPALS} natural principals"
        ));
    }
    if document.privileged_routes.len() > MAX_PRIVILEGED_ROUTES {
        return Err(format!(
            "request-class catalog supports at most {MAX_PRIVILEGED_ROUTES} privileged routes"
        ));
    }

    let mut natural_principals = BTreeMap::new();
    let mut workload_buckets = BTreeSet::new();
    for (principal, bucket) in document.natural_principals {
        validate_authenticated_principal(&principal)?;
        validate_identity("workload_bucket", &bucket)?;
        if bucket == UNASSIGNED {
            return Err("catalogued workload_bucket cannot be unassigned".to_string());
        }
        workload_buckets.insert(bucket.clone());
        natural_principals.insert(principal, Arc::from(bucket));
    }

    let mut privileged_routes = BTreeMap::new();
    let mut catalog_cache_domains = BTreeSet::new();
    for (route, policy) in document.privileged_routes {
        validate_identity("trusted_route_identity", &route)?;
        validate_authenticated_principal(&policy.authenticated_principal)?;
        validate_http_endpoint(&policy.endpoint)?;
        validate_identity("workload_bucket", &policy.workload_bucket)?;
        if policy.workload_bucket == UNASSIGNED {
            return Err("catalogued workload_bucket cannot be unassigned".to_string());
        }
        workload_buckets.insert(policy.workload_bucket.clone());
        validate_identity("pd_service", &policy.pd_service)?;
        validate_identity("smg_release", &policy.smg_release)?;
        validate_identity("release_generation", &policy.release_generation)?;
        validate_digest("membership_checksum", &policy.membership_checksum, false)?;
        if policy.cache_domains.is_empty()
            || policy.cache_domains.len() > MAX_CACHE_DOMAINS_PER_ROUTE
        {
            return Err(format!(
                "privileged route cache_domains must contain 1..={MAX_CACHE_DOMAINS_PER_ROUTE} values"
            ));
        }
        let mut cache_domains = BTreeSet::new();
        for cache_domain in policy.cache_domains {
            validate_identity("cache_domain", &cache_domain)?;
            if cache_domain == UNASSIGNED {
                return Err("catalogued cache_domain cannot be unassigned".to_string());
            }
            if !cache_domains.insert(Arc::from(cache_domain.as_str())) {
                return Err(
                    "privileged route cache_domains must not contain duplicates".to_string()
                );
            }
            catalog_cache_domains.insert(cache_domain);
        }
        if catalog_cache_domains.len() > MAX_CATALOG_CACHE_DOMAINS {
            return Err(format!(
                "request-class catalog supports at most {MAX_CATALOG_CACHE_DOMAINS} cache domains"
            ));
        }
        let traffic_class = match policy.traffic_class.as_str() {
            "synthetic" => TrafficClass::Synthetic,
            "benchmark" => TrafficClass::Benchmark,
            _ => {
                return Err(
                    "privileged route traffic_class must be synthetic or benchmark".to_string(),
                )
            }
        };
        privileged_routes.insert(
            route,
            PrivilegedRoutePolicy {
                authenticated_principal: Arc::from(policy.authenticated_principal),
                endpoint: Arc::from(policy.endpoint),
                traffic_class,
                workload_bucket: Arc::from(policy.workload_bucket),
                cache_domains,
                response_attribution: PdResponseAttribution {
                    environment: Arc::from(UNASSIGNED),
                    traffic_class,
                    pd_service: Arc::from(policy.pd_service),
                    smg_release: Arc::from(policy.smg_release),
                    release_generation: Arc::from(policy.release_generation),
                    membership_checksum: Arc::from(policy.membership_checksum),
                    cache_domain: Arc::from(UNASSIGNED),
                },
            },
        );
    }
    if workload_buckets.len() > MAX_WORKLOAD_BUCKETS {
        return Err(format!(
            "request-class catalog supports at most {MAX_WORKLOAD_BUCKETS} workload buckets"
        ));
    }

    Ok(RequestClassCatalog {
        version: Arc::from(document.version),
        natural_principals,
        privileged_routes,
    })
}

fn parse_cohort_catalog(json: &str) -> Result<CohortCatalog, String> {
    let document: CohortCatalogDocument = serde_json::from_str(json)
        .map_err(|error| format!("invalid cohort catalog JSON: {error}"))?;
    if document.version != COHORT_CATALOG_VERSION {
        return Err(format!(
            "cohort catalog version must be {COHORT_CATALOG_VERSION}"
        ));
    }
    if document.pairs.is_empty() {
        return Err("cohort catalog must contain at least one pair".to_string());
    }
    if document.pairs.len() > MAX_COHORT_PAIRS {
        return Err(format!(
            "cohort catalog supports at most {MAX_COHORT_PAIRS} pairs"
        ));
    }
    for (pair_id, pair) in &document.pairs {
        validate_identity("execution_cohort_pair_id", pair_id)?;
        validate_identity("cache_domain", &pair.cache_domain)?;
        validate_identity("prefill_runtime_cohort", &pair.prefill_runtime_cohort)?;
        validate_identity("decode_runtime_cohort", &pair.decode_runtime_cohort)?;
        if [
            pair_id.as_str(),
            pair.cache_domain.as_str(),
            pair.prefill_runtime_cohort.as_str(),
            pair.decode_runtime_cohort.as_str(),
        ]
        .contains(&UNASSIGNED)
        {
            return Err("cohort catalog values cannot be unassigned".to_string());
        }
    }
    Ok(CohortCatalog {
        version: Arc::from(document.version),
        pairs: document.pairs,
    })
}

fn validate_authenticated_principal(principal: &str) -> Result<(), String> {
    let Some(digest) = principal.strip_prefix("auth:") else {
        return Err(
            "request-class catalog principals must be authenticated auth:<sha256> keys".to_string(),
        );
    };
    if digest.len() == 64
        && digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(
            "request-class catalog principals must contain a 64-character lowercase SHA-256 digest"
                .to_string(),
        )
    }
}

fn validate_http_endpoint(endpoint: &str) -> Result<(), String> {
    if endpoint.starts_with('/')
        && endpoint.len() <= 128
        && endpoint
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'/' | b'.' | b'_' | b'-'))
    {
        Ok(())
    } else {
        Err("request-class catalog endpoint must be a bounded absolute HTTP path".to_string())
    }
}

fn trusted_probe_cache_domain(headers: Option<&HeaderMap>) -> Result<&str, String> {
    let headers = headers.ok_or_else(|| "trusted probe cache domain is required".to_string())?;
    let mut values = headers.get_all(PROBE_CACHE_DOMAIN_HEADER).iter();
    let value = values
        .next()
        .ok_or_else(|| "trusted probe cache domain is required".to_string())?;
    if values.next().is_some() {
        return Err("trusted probe cache domain must appear exactly once".to_string());
    }
    let value = value
        .to_str()
        .map_err(|_| "trusted probe cache domain must be ASCII".to_string())?;
    validate_identity("trusted probe cache domain", value)?;
    Ok(value)
}

fn valid_worker_label<'a>(
    labels: &'a std::collections::HashMap<String, String>,
    key: &str,
) -> Option<&'a str> {
    labels
        .get(key)
        .filter(|value| validate_identity(key, value).is_ok())
        .map(String::as_str)
}

fn validate_environment(environment: &str) -> Result<(), String> {
    if matches!(environment, "test" | "prod") {
        Ok(())
    } else {
        Err("environment must be test or prod".to_string())
    }
}

fn validate_identity(field: &str, value: &str) -> Result<(), String> {
    let valid = !value.is_empty()
        && value.len() <= MAX_IDENTITY_LEN
        && value.as_bytes()[0].is_ascii_alphanumeric()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-' | b':'));
    if valid {
        Ok(())
    } else {
        Err(format!(
            "{field} must be 1..={MAX_IDENTITY_LEN} ASCII identity characters"
        ))
    }
}

fn validate_digest(field: &str, value: &str, require_prefix: bool) -> Result<(), String> {
    let digest = if require_prefix {
        value
            .strip_prefix("sha256:")
            .ok_or_else(|| format!("{field} must use the sha256:<64 lowercase hex> form"))?
    } else {
        value
    };
    if digest.len() == 64
        && digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(format!(
            "{field} must contain exactly 64 lowercase hex characters"
        ))
    }
}

fn validate_revision(value: &str) -> Result<(), String> {
    if matches!(value.len(), 40 | 64)
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(
            "producer_revision must be an exact 40 or 64 character lowercase git revision"
                .to_string(),
        )
    }
}

/// Server-assigned request class. Public HTTP is natural until an authenticated
/// route identity is added to `TenantRequestMeta`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TrafficClass {
    Natural,
    Synthetic,
    Benchmark,
}

impl TrafficClass {
    fn as_str(self) -> &'static str {
        match self {
            Self::Natural => "natural",
            Self::Synthetic => "synthetic",
            Self::Benchmark => "benchmark",
        }
    }
}

/// Server-owned identity for an isolated ingress route. Public headers never
/// construct this value; a trusted authentication/routing middleware must add
/// it to [`TenantRequestMeta`] after validating the dedicated credential.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PdTrustedRouteIdentity(Arc<str>);

impl PdTrustedRouteIdentity {
    pub fn new(route_identity: &str) -> Result<Self, String> {
        validate_identity("trusted_route_identity", route_identity)?;
        Ok(Self(Arc::from(route_identity)))
    }

    fn as_str(&self) -> &str {
        &self.0
    }
}

/// Low-cardinality classification resolved entirely from server-owned state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PdRequestClassification {
    traffic_class: TrafficClass,
    workload_bucket: Arc<str>,
    response_attribution: Option<PdResponseAttribution>,
}

/// Stable server-side values attached by an opt-in dedicated probe/harness
/// route. They are not serialized by public serving handlers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PdResponseAttribution {
    environment: Arc<str>,
    traffic_class: TrafficClass,
    pd_service: Arc<str>,
    smg_release: Arc<str>,
    release_generation: Arc<str>,
    membership_checksum: Arc<str>,
    cache_domain: Arc<str>,
}

impl PdResponseAttribution {
    pub fn environment(&self) -> &str {
        &self.environment
    }

    pub fn traffic_class(&self) -> &'static str {
        self.traffic_class.as_str()
    }

    pub fn pd_service(&self) -> &str {
        &self.pd_service
    }

    pub fn smg_release(&self) -> &str {
        &self.smg_release
    }

    pub fn release_generation(&self) -> &str {
        &self.release_generation
    }

    pub fn membership_checksum(&self) -> &str {
        &self.membership_checksum
    }

    pub fn cache_domain(&self) -> &str {
        &self.cache_domain
    }

    /// Serialize trusted attribution for the dedicated internal probe handler.
    /// Public serving handlers never call this method.
    pub(crate) fn to_internal_probe_headers(&self) -> Result<HeaderMap, InvalidHeaderValue> {
        let values = [
            (
                PROBE_RESPONSE_SERVICE_HEADER,
                HeaderValue::from_str(self.pd_service())?,
            ),
            (
                PROBE_RESPONSE_RELEASE_HEADER,
                HeaderValue::from_str(self.smg_release())?,
            ),
            (
                PROBE_RESPONSE_ENVIRONMENT_HEADER,
                HeaderValue::from_str(self.environment())?,
            ),
            (
                PROBE_RESPONSE_RELEASE_GENERATION_HEADER,
                HeaderValue::from_str(self.release_generation())?,
            ),
            (
                PROBE_RESPONSE_MEMBERSHIP_CHECKSUM_HEADER,
                HeaderValue::from_str(self.membership_checksum())?,
            ),
            (
                PROBE_RESPONSE_CACHE_DOMAIN_HEADER,
                HeaderValue::from_str(self.cache_domain())?,
            ),
            (
                PROBE_RESPONSE_TRAFFIC_CLASS_HEADER,
                HeaderValue::from_str(self.traffic_class())?,
            ),
        ];
        let mut headers = HeaderMap::with_capacity(values.len());
        for (name, value) in values {
            headers.insert(name, value);
        }
        Ok(headers)
    }

    /// Prevent a Worker response from exposing internal probe attribution on
    /// the public serving surface.
    pub(crate) fn strip_internal_probe_headers(headers: &mut HeaderMap) {
        for name in PROBE_RESPONSE_HEADERS {
            headers.remove(name);
        }
    }
}

/// Low-cardinality identity selected for the final PD attempt.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PdExecutionIdentity {
    workload_bucket: Arc<str>,
    cache_domain: Arc<str>,
    prefill_runtime_cohort: Arc<str>,
    decode_runtime_cohort: Arc<str>,
    execution_cohort_pair_id: Arc<str>,
}

impl PdExecutionIdentity {
    /// Construct a validated identity for tests and non-worker producers.
    pub fn new(
        workload_bucket: &str,
        cache_domain: &str,
        prefill_runtime_cohort: &str,
        decode_runtime_cohort: &str,
        execution_cohort_pair_id: &str,
    ) -> Result<Self, String> {
        for (field, value) in [
            ("workload_bucket", workload_bucket),
            ("cache_domain", cache_domain),
            ("prefill_runtime_cohort", prefill_runtime_cohort),
            ("decode_runtime_cohort", decode_runtime_cohort),
            ("execution_cohort_pair_id", execution_cohort_pair_id),
        ] {
            validate_identity(field, value)?;
        }
        Ok(Self {
            workload_bucket: Arc::from(workload_bucket),
            cache_domain: Arc::from(cache_domain),
            prefill_runtime_cohort: Arc::from(prefill_runtime_cohort),
            decode_runtime_cohort: Arc::from(decode_runtime_cohort),
            execution_cohort_pair_id: Arc::from(execution_cohort_pair_id),
        })
    }

    fn unassigned() -> Self {
        Self {
            workload_bucket: Arc::from(UNASSIGNED),
            cache_domain: Arc::from(UNASSIGNED),
            prefill_runtime_cohort: Arc::from(UNASSIGNED),
            decode_runtime_cohort: Arc::from(UNASSIGNED),
            execution_cohort_pair_id: Arc::from(UNASSIGNED),
        }
    }
}

/// Final service outcome, independent of retry attempts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PdTerminalOutcome {
    Success,
    Error,
    Cancel,
}

impl PdTerminalOutcome {
    fn as_str(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::Error => "error",
            Self::Cancel => "cancel",
        }
    }
}

/// Validated final request usage. Both values must be present together.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PdUsage {
    input_tokens: u64,
    output_tokens: u64,
}

impl PdUsage {
    pub fn new(input_tokens: u64, output_tokens: u64) -> Self {
        Self {
            input_tokens,
            output_tokens,
        }
    }
}

#[derive(Debug)]
struct RequestObservation {
    identity: PdExecutionIdentity,
    first_output_token: Option<Duration>,
    last_output_token: Option<Duration>,
    output_events: u64,
    choice_output_events: BTreeMap<u64, u64>,
    timing_ambiguous: bool,
    observation_overflowed: bool,
    reported_output_tokens: u64,
    usage: Option<PdUsage>,
}

impl RequestObservation {
    fn new_for_workload(workload_bucket: Arc<str>) -> Self {
        let mut identity = PdExecutionIdentity::unassigned();
        identity.workload_bucket = workload_bucket;
        Self {
            identity,
            first_output_token: None,
            last_output_token: None,
            output_events: 0,
            choice_output_events: BTreeMap::new(),
            timing_ambiguous: false,
            observation_overflowed: false,
            reported_output_tokens: 0,
            usage: None,
        }
    }
}

/// Exactly-once service request state shared across all retry attempts and,
/// for streaming responses, the body relay task.
#[derive(Debug)]
pub struct PdRequestLifecycle {
    config: Arc<PdLifecycleConfig>,
    traffic_class: TrafficClass,
    workload_bucket: Arc<str>,
    response_attribution: Option<PdResponseAttribution>,
    started_at: Instant,
    terminal: AtomicBool,
    observation: Mutex<RequestObservation>,
}

impl PdRequestLifecycle {
    /// Start one service request and increment the started counter once.
    #[cfg(test)]
    pub fn start(config: Arc<PdLifecycleConfig>, traffic_class: TrafficClass) -> Arc<Self> {
        Self::start_classified(
            config,
            PdRequestClassification {
                traffic_class,
                workload_bucket: Arc::from(UNASSIGNED),
                response_attribution: None,
            },
        )
    }

    pub fn start_classified(
        config: Arc<PdLifecycleConfig>,
        classification: PdRequestClassification,
    ) -> Arc<Self> {
        Metrics::record_pd_request_started(&config, classification.traffic_class);
        Arc::new(Self {
            observation: Mutex::new(RequestObservation::new_for_workload(Arc::clone(
                &classification.workload_bucket,
            ))),
            config,
            traffic_class: classification.traffic_class,
            workload_bucket: classification.workload_bucket,
            response_attribution: classification.response_attribution,
            started_at: Instant::now(),
            terminal: AtomicBool::new(false),
        })
    }

    /// Begin an attempt. This discards observations from a retryable previous
    /// attempt so only the final attempt contributes terminal evidence.
    pub fn begin_attempt(&self, mut identity: PdExecutionIdentity) {
        let mut observation = self
            .observation
            .lock()
            .unwrap_or_else(|lock| lock.into_inner());
        if self.terminal.load(Ordering::Acquire) {
            return;
        }
        identity.workload_bucket = Arc::clone(&self.workload_bucket);
        *observation = RequestObservation {
            identity,
            first_output_token: None,
            last_output_token: None,
            output_events: 0,
            choice_output_events: BTreeMap::new(),
            timing_ambiguous: false,
            observation_overflowed: false,
            reported_output_tokens: 0,
            usage: None,
        };
    }

    /// Begin an attempt using identity carried by the selected workers.
    pub fn begin_selected_attempt(
        &self,
        prefill: &dyn Worker,
        decode: &dyn Worker,
    ) -> Result<(), String> {
        let identity = self.config.execution_identity(prefill, decode);
        if let Some(attribution) = self.response_attribution.as_ref() {
            if identity.cache_domain.as_ref() != attribution.cache_domain.as_ref() {
                return Err(
                    "selected PD pair does not match the privileged route cache domain".to_string(),
                );
            }
        }
        self.begin_attempt(identity);
        Ok(())
    }

    pub fn response_attribution(&self) -> Option<&PdResponseAttribution> {
        self.response_attribution.as_ref()
    }

    /// Reset to the explicit pre-dispatch identity before worker selection.
    pub fn begin_unassigned_attempt(&self) {
        let mut observation = self
            .observation
            .lock()
            .unwrap_or_else(|lock| lock.into_inner());
        if self.terminal.load(Ordering::Acquire) {
            return;
        }
        *observation = RequestObservation::new_for_workload(Arc::clone(&self.workload_bucket));
    }

    /// Observe a decoded JSON response event at the current monotonic time.
    pub fn observe_json(&self, value: &Value) {
        self.observe_json_after(value, self.started_at.elapsed());
    }

    /// Observe a complete non-streaming response. A buffered body can provide
    /// final usage, but its arrival cannot prove when the first token was
    /// emitted, so it must not manufacture TTFT or TPOT evidence.
    pub fn observe_non_stream_json(&self, value: &Value) {
        let Some(usage) = parse_usage(value) else {
            return;
        };
        let mut observation = self
            .observation
            .lock()
            .unwrap_or_else(|lock| lock.into_inner());
        if !self.terminal.load(Ordering::Acquire) {
            observation.usage = Some(usage);
        }
    }

    fn observe_json_after(&self, value: &Value, elapsed: Duration) {
        let usage = parse_usage(value);
        let reported_output_tokens = output_token_count(value);
        let output_identity = output_event_identity(value);
        let mut observation = self
            .observation
            .lock()
            .unwrap_or_else(|lock| lock.into_inner());
        if self.terminal.load(Ordering::Acquire) || observation.observation_overflowed {
            return;
        }

        if let Some(usage) = usage {
            observation.usage = Some(usage);
        }

        let output_advanced = reported_output_tokens
            .filter(|count| *count > observation.reported_output_tokens)
            .is_some();
        if let Some(count) = reported_output_tokens {
            observation.reported_output_tokens = observation.reported_output_tokens.max(count);
        }
        match output_identity {
            OutputEventIdentity::None if output_advanced && value.get("choices").is_none() => {
                observe_output_token(&mut observation, elapsed);
            }
            OutputEventIdentity::None => {}
            OutputEventIdentity::LegacySingle => {
                observe_output_token(&mut observation, elapsed);
            }
            OutputEventIdentity::UnindexedChoice => {
                observation.timing_ambiguous = true;
                observe_output_token(&mut observation, elapsed);
            }
            OutputEventIdentity::IndexedChoices(indices) => {
                for index in indices {
                    if let Some(events) = observation.choice_output_events.get_mut(&index) {
                        *events = events.saturating_add(1);
                    } else if observation.choice_output_events.len() < MAX_TRACKED_CHOICE_INDICES {
                        observation.choice_output_events.insert(index, 1);
                    } else {
                        observation.timing_ambiguous = true;
                    }
                }
                observe_output_token(&mut observation, elapsed);
            }
        }
    }

    /// Stop accepting partial observations after a bounded parser limit is
    /// exceeded. The response bytes continue downstream unchanged, while all
    /// token-derived evidence for this attempt remains explicitly unknown.
    pub(crate) fn mark_observation_overflowed(&self) {
        let mut observation = self
            .observation
            .lock()
            .unwrap_or_else(|lock| lock.into_inner());
        if self.terminal.load(Ordering::Acquire) {
            return;
        }
        observation.observation_overflowed = true;
        observation.timing_ambiguous = true;
        observation.usage = None;
    }

    #[cfg(test)]
    fn observe_output_token_after(&self, elapsed: Duration) {
        let mut observation = self
            .observation
            .lock()
            .unwrap_or_else(|lock| lock.into_inner());
        if self.terminal.load(Ordering::Acquire) {
            return;
        }
        observe_output_token(&mut observation, elapsed);
    }

    /// Finish exactly once. A second caller is a no-op and returns false.
    pub fn finish(
        &self,
        outcome: PdTerminalOutcome,
        status: StatusCode,
        usage: Option<PdUsage>,
    ) -> bool {
        if self
            .terminal
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return false;
        }

        let mut observation = self
            .observation
            .lock()
            .unwrap_or_else(|lock| lock.into_inner());
        if observation.observation_overflowed {
            observation.usage = None;
        } else if usage.is_some() {
            observation.usage = usage;
        } else if outcome != PdTerminalOutcome::Success {
            // Stream events can contain cumulative/intermediate usage. Without
            // a successful terminal event that is not complete final usage,
            // so cancellations and partial-stream errors fail closed.
            observation.usage = None;
        }
        Metrics::record_pd_request_terminal(
            &self.config,
            self.traffic_class,
            outcome,
            status,
            self.started_at.elapsed(),
            &observation,
        );
        true
    }

    /// Finish using any usage observed in the streamed response.
    pub fn finish_stream(&self, outcome: PdTerminalOutcome, status: StatusCode) -> bool {
        self.finish(outcome, status, None)
    }
}

impl Drop for PdRequestLifecycle {
    fn drop(&mut self) {
        // The last lifecycle owner disappears when a request future is dropped
        // before it can build a response (for example, scheduler preemption).
        // Use one fixed status class for this bounded cancellation reason. The
        // terminal CAS also prevents a later stream/body guard from double
        // counting a request that already reached a terminal state.
        self.finish_stream(PdTerminalOutcome::Cancel, StatusCode::REQUEST_TIMEOUT);
    }
}

fn observe_output_token(observation: &mut RequestObservation, elapsed: Duration) {
    observation.first_output_token.get_or_insert(elapsed);
    observation.last_output_token = Some(elapsed);
    observation.output_events = observation.output_events.saturating_add(1);
}

/// Incremental, bounded SSE observer. It never changes response bytes.
#[cfg(test)]
pub struct PdSseObserver {
    request: Arc<PdRequestLifecycle>,
    parser: PdSseParser,
}

/// Result of feeding one raw chunk into the incremental SSE parser.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PdSseParseState {
    pub done: bool,
    pub application_error_seen: bool,
    /// At least one SSE line or assembled data event exceeded the fixed
    /// observation buffer. Response forwarding remains unaffected, but token
    /// evidence derived from this parser is incomplete.
    pub buffer_overflowed: bool,
}

/// One bounded incremental SSE parser shared by terminal detection and metric
/// observation. It avoids chunk-local substring scans and per-event join/drain
/// allocations while preserving split-event correctness.
#[derive(Default)]
pub struct PdSseParser {
    line: Vec<u8>,
    data: Vec<u8>,
    event_overflowed: bool,
    buffer_overflowed: bool,
    done: bool,
    application_error_seen: bool,
}

/// RAII cancellation signal tied to the downstream response body. Normal
/// stream completion wins the lifecycle's exactly-once terminal race; dropping
/// the body first records cancellation even if the upstream is stalled.
pub struct PdStreamCancellationGuard {
    request: Option<Arc<PdRequestLifecycle>>,
    status: StatusCode,
}

impl PdStreamCancellationGuard {
    pub fn new(request: Option<Arc<PdRequestLifecycle>>, status: StatusCode) -> Self {
        Self { request, status }
    }
}

impl Drop for PdStreamCancellationGuard {
    fn drop(&mut self) {
        if let Some(request) = self.request.take() {
            request.finish_stream(PdTerminalOutcome::Cancel, self.status);
        }
    }
}

#[cfg(test)]
impl PdSseObserver {
    pub fn new(request: Arc<PdRequestLifecycle>) -> Self {
        Self {
            request,
            parser: PdSseParser::default(),
        }
    }

    /// Observe one upstream chunk while bounding diagnostic memory.
    pub fn observe_chunk(&mut self, chunk: &[u8]) {
        let request = Arc::clone(&self.request);
        self.parser
            .observe_chunk(chunk, |value| request.observe_json(&value));
    }
}

impl PdSseParser {
    pub fn has_pending_event(&self) -> bool {
        !self.line.is_empty() || !self.data.is_empty() || self.event_overflowed
    }

    pub fn observe_chunk(
        &mut self,
        chunk: &[u8],
        mut observe_json: impl FnMut(Value),
    ) -> PdSseParseState {
        if self.done {
            return self.state();
        }
        for byte in chunk {
            if *byte == b'\n' {
                let line = self.line.strip_suffix(b"\r").unwrap_or(&self.line);
                if line.is_empty() {
                    if !self.event_overflowed && self.data == b"[DONE]" {
                        self.done = true;
                    } else if !self.event_overflowed {
                        if let Ok(value) = serde_json::from_slice::<Value>(&self.data) {
                            if value.get("error").is_some() {
                                self.application_error_seen = true;
                            }
                            observe_json(value);
                        }
                    }
                    self.data.clear();
                    self.event_overflowed = false;
                } else if let Some(data) = line.strip_prefix(b"data:") {
                    let data = data.strip_prefix(b" ").unwrap_or(data);
                    let separator = usize::from(!self.data.is_empty());
                    if self
                        .data
                        .len()
                        .saturating_add(separator)
                        .saturating_add(data.len())
                        <= MAX_SSE_OBSERVER_BUFFER
                    {
                        if separator == 1 {
                            self.data.push(b'\n');
                        }
                        self.data.extend_from_slice(data);
                    } else {
                        self.event_overflowed = true;
                        self.buffer_overflowed = true;
                    }
                }
                self.line.clear();
                if self.done {
                    break;
                }
            } else if self.line.len() < MAX_SSE_OBSERVER_BUFFER {
                self.line.push(*byte);
            } else {
                self.event_overflowed = true;
                self.buffer_overflowed = true;
            }
        }
        self.state()
    }

    fn state(&self) -> PdSseParseState {
        PdSseParseState {
            done: self.done,
            application_error_seen: self.application_error_seen,
            buffer_overflowed: self.buffer_overflowed,
        }
    }
}

fn parse_usage(value: &Value) -> Option<PdUsage> {
    let candidates = [
        ("/usage/prompt_tokens", "/usage/completion_tokens"),
        ("/usage/input_tokens", "/usage/output_tokens"),
        ("/meta_info/prompt_tokens", "/meta_info/completion_tokens"),
    ];
    candidates.iter().find_map(|(input_path, output_path)| {
        Some(PdUsage::new(
            value.pointer(input_path)?.as_u64()?,
            value.pointer(output_path)?.as_u64()?,
        ))
    })
}

fn output_token_count(value: &Value) -> Option<u64> {
    [
        "/meta_info/completion_tokens",
        "/usage/completion_tokens",
        "/usage/output_tokens",
    ]
    .iter()
    .find_map(|path| value.pointer(path).and_then(Value::as_u64))
}

enum OutputEventIdentity {
    None,
    LegacySingle,
    UnindexedChoice,
    IndexedChoices(Vec<u64>),
}

fn output_event_identity(value: &Value) -> OutputEventIdentity {
    if value
        .pointer("/text")
        .and_then(Value::as_str)
        .is_some_and(|text| !text.is_empty())
    {
        return OutputEventIdentity::LegacySingle;
    }

    let Some(choices) = value.get("choices").and_then(Value::as_array) else {
        return OutputEventIdentity::None;
    };
    let mut indices = Vec::new();
    for choice in choices {
        let has_output = choice
            .get("text")
            .and_then(Value::as_str)
            .is_some_and(|text| !text.is_empty())
            || choice
                .pointer("/delta/content")
                .and_then(Value::as_str)
                .is_some_and(|text| !text.is_empty())
            || choice
                .pointer("/delta/tool_calls")
                .and_then(Value::as_array)
                .is_some_and(|tool_calls| !tool_calls.is_empty())
            || choice
                .pointer("/delta/reasoning_content")
                .and_then(Value::as_str)
                .is_some_and(|reasoning| !reasoning.is_empty());
        if !has_output {
            continue;
        }
        let Some(index) = choice.get("index").and_then(Value::as_u64) else {
            return OutputEventIdentity::UnindexedChoice;
        };
        if !indices.contains(&index) {
            indices.push(index);
        }
    }
    if indices.is_empty() {
        OutputEventIdentity::None
    } else {
        OutputEventIdentity::IndexedChoices(indices)
    }
}

fn status_class(status: StatusCode) -> &'static str {
    match status.as_u16() / 100 {
        1 => "1xx",
        2 => "2xx",
        3 => "3xx",
        4 => "4xx",
        5 => "5xx",
        _ => "other",
    }
}

impl Metrics {
    fn install_pd_lifecycle_capability(config: &PdLifecycleConfig) {
        let request_class_catalog_version = config
            .request_class_catalog
            .as_ref()
            .map_or(UNASSIGNED, |catalog| catalog.version.as_ref());
        let cohort_catalog_version = config
            .cohort_catalog
            .as_ref()
            .map_or(UNASSIGNED, |catalog| catalog.version.as_ref());
        gauge!(
            "smg_pd_request_lifecycle_contract_info",
            "environment" => Arc::clone(&config.environment),
            "pd_service" => Arc::clone(&config.pd_service),
            "smg_release" => Arc::clone(&config.smg_release),
            "metric_contract_version" => Arc::clone(&config.metric_contract_version),
            "schema_digest" => Arc::clone(&config.schema_digest),
            "image_digest" => Arc::clone(&config.image_digest),
            "producer_revision" => Arc::clone(&config.producer_revision),
            "producer_transport" => "http",
            "request_class_catalog_version" => request_class_catalog_version.to_string(),
            "cohort_catalog_version" => cohort_catalog_version.to_string(),
        )
        .set(1.0);
    }

    fn record_pd_request_started(config: &PdLifecycleConfig, traffic_class: TrafficClass) {
        counter!(
            "smg_pd_requests_started_total",
            "environment" => Arc::clone(&config.environment),
            "pd_service" => Arc::clone(&config.pd_service),
            "smg_release" => Arc::clone(&config.smg_release),
            "traffic_class" => traffic_class.as_str(),
        )
        .increment(1);
    }

    fn record_pd_request_terminal(
        config: &PdLifecycleConfig,
        traffic_class: TrafficClass,
        outcome: PdTerminalOutcome,
        status: StatusCode,
        e2e: Duration,
        observation: &RequestObservation,
    ) {
        let identity = &observation.identity;
        let status_class = status_class(status);
        counter!(
            "smg_pd_terminal_requests_total",
            "environment" => Arc::clone(&config.environment),
            "pd_service" => Arc::clone(&config.pd_service),
            "smg_release" => Arc::clone(&config.smg_release),
            "traffic_class" => traffic_class.as_str(),
            "workload_bucket" => Arc::clone(&identity.workload_bucket),
            "cache_domain" => Arc::clone(&identity.cache_domain),
            "outcome" => outcome.as_str(),
            "status_class" => status_class,
            "prefill_runtime_cohort" => Arc::clone(&identity.prefill_runtime_cohort),
            "decode_runtime_cohort" => Arc::clone(&identity.decode_runtime_cohort),
            "execution_cohort_pair_id" => Arc::clone(&identity.execution_cohort_pair_id),
        )
        .increment(1);

        Self::record_pd_evidence(config, traffic_class, outcome, identity, "e2e", true);
        Self::record_pd_histogram(
            "smg_pd_e2e_seconds",
            config,
            traffic_class,
            outcome,
            identity,
            e2e.as_secs_f64(),
        );

        let ttft = observation
            .first_output_token
            .filter(|_| !observation.timing_ambiguous);
        Self::record_pd_evidence(
            config,
            traffic_class,
            outcome,
            identity,
            "ttft",
            ttft.is_some(),
        );
        if let Some(ttft) = ttft {
            Self::record_pd_histogram(
                "smg_pd_request_v1_ttft_seconds",
                config,
                traffic_class,
                outcome,
                identity,
                ttft.as_secs_f64(),
            );
        }

        let usage_known = observation.usage.is_some();
        Self::record_pd_evidence(
            config,
            traffic_class,
            outcome,
            identity,
            "usage",
            usage_known,
        );
        if let Some(usage) = observation.usage {
            Self::record_pd_tokens(config, traffic_class, outcome, identity, usage);
        }

        let tpot = match (
            traffic_class,
            outcome,
            observation.usage,
            observation.first_output_token,
            observation.last_output_token,
        ) {
            (
                TrafficClass::Natural,
                PdTerminalOutcome::Success,
                Some(usage),
                Some(first),
                Some(last),
            ) if usage.output_tokens >= 2
                && observation.output_events >= 2
                && usage.output_tokens >= observation.output_events
                && !observation.timing_ambiguous
                && last >= first =>
            {
                Some((last - first).as_secs_f64() / (usage.output_tokens - 1) as f64)
            }
            _ => None,
        };
        Self::record_pd_evidence(
            config,
            traffic_class,
            outcome,
            identity,
            "tpot",
            tpot.is_some(),
        );
        if let Some(tpot) = tpot {
            Self::record_pd_histogram(
                "smg_pd_tpot_seconds",
                config,
                traffic_class,
                outcome,
                identity,
                tpot,
            );
        }
    }

    fn record_pd_evidence(
        config: &PdLifecycleConfig,
        traffic_class: TrafficClass,
        outcome: PdTerminalOutcome,
        identity: &PdExecutionIdentity,
        metric: &'static str,
        known: bool,
    ) {
        counter!(
            "smg_pd_request_metric_evidence_total",
            "environment" => Arc::clone(&config.environment),
            "pd_service" => Arc::clone(&config.pd_service),
            "smg_release" => Arc::clone(&config.smg_release),
            "traffic_class" => traffic_class.as_str(),
            "workload_bucket" => Arc::clone(&identity.workload_bucket),
            "cache_domain" => Arc::clone(&identity.cache_domain),
            "outcome" => outcome.as_str(),
            "prefill_runtime_cohort" => Arc::clone(&identity.prefill_runtime_cohort),
            "decode_runtime_cohort" => Arc::clone(&identity.decode_runtime_cohort),
            "execution_cohort_pair_id" => Arc::clone(&identity.execution_cohort_pair_id),
            "metric" => metric,
            "evidence_state" => if known { "known" } else { "unknown" },
        )
        .increment(1);
    }

    fn record_pd_histogram(
        name: &'static str,
        config: &PdLifecycleConfig,
        traffic_class: TrafficClass,
        outcome: PdTerminalOutcome,
        identity: &PdExecutionIdentity,
        value: f64,
    ) {
        histogram!(
            name,
            "environment" => Arc::clone(&config.environment),
            "pd_service" => Arc::clone(&config.pd_service),
            "smg_release" => Arc::clone(&config.smg_release),
            "traffic_class" => traffic_class.as_str(),
            "workload_bucket" => Arc::clone(&identity.workload_bucket),
            "cache_domain" => Arc::clone(&identity.cache_domain),
            "outcome" => outcome.as_str(),
            "prefill_runtime_cohort" => Arc::clone(&identity.prefill_runtime_cohort),
            "decode_runtime_cohort" => Arc::clone(&identity.decode_runtime_cohort),
            "execution_cohort_pair_id" => Arc::clone(&identity.execution_cohort_pair_id),
        )
        .record(value);
    }

    fn record_pd_tokens(
        config: &PdLifecycleConfig,
        traffic_class: TrafficClass,
        outcome: PdTerminalOutcome,
        identity: &PdExecutionIdentity,
        usage: PdUsage,
    ) {
        for (name, tokens) in [
            ("smg_pd_completed_input_tokens_total", usage.input_tokens),
            ("smg_pd_completed_output_tokens_total", usage.output_tokens),
        ] {
            counter!(
                name,
                "environment" => Arc::clone(&config.environment),
                "pd_service" => Arc::clone(&config.pd_service),
                "smg_release" => Arc::clone(&config.smg_release),
                "traffic_class" => traffic_class.as_str(),
                "workload_bucket" => Arc::clone(&identity.workload_bucket),
                "cache_domain" => Arc::clone(&identity.cache_domain),
                "outcome" => outcome.as_str(),
                "prefill_runtime_cohort" => Arc::clone(&identity.prefill_runtime_cohort),
                "decode_runtime_cohort" => Arc::clone(&identity.decode_runtime_cohort),
                "execution_cohort_pair_id" => Arc::clone(&identity.execution_cohort_pair_id),
            )
            .increment(tokens);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc, time::Duration};

    use axum::http::{HeaderMap, HeaderValue, StatusCode};
    use metrics_exporter_prometheus::PrometheusBuilder;

    use super::{
        parse_request_class_catalog, parse_usage, PdExecutionIdentity, PdLifecycleConfig,
        PdRequestLifecycle, PdResponseAttribution, PdSseObserver, PdStreamCancellationGuard,
        PdTerminalOutcome, PdTrustedRouteIdentity, PdUsage, TrafficClass,
        PROBE_CACHE_DOMAIN_HEADER, UNASSIGNED,
    };
    use crate::tenant::{RouteRequestMeta, TenantKey};

    fn config() -> Arc<PdLifecycleConfig> {
        config_for_environment("test")
    }

    fn config_for_environment(environment: &str) -> Arc<PdLifecycleConfig> {
        Arc::new(
            PdLifecycleConfig::new(
                environment,
                "glm52",
                "stable",
                "smg-pd-request-v1",
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "0744ea180744ea180744ea180744ea180744ea18",
            )
            .expect("valid fixture"),
        )
    }

    fn identity() -> PdExecutionIdentity {
        PdExecutionIdentity::new("agent-80k-v1", "cache-a", "p-v1", "d-v2", "p-v1_d-v2")
            .expect("valid fixture")
    }

    fn config_with_catalogs() -> Arc<PdLifecycleConfig> {
        let principal = format!("auth:{}", "1".repeat(64));
        let request_catalog = serde_json::json!({
            "version": "request-class-v1",
            "natural_principals": {
                principal.clone(): "agent-80k-v1"
            },
            "privileged_routes": {
                "internal-probe-v1": {
                    "authenticated_principal": principal,
                    "endpoint": "/v1/chat/completions",
                    "traffic_class": "synthetic",
                    "workload_bucket": "probe-2k-v1",
                    "pd_service": "glm52",
                    "smg_release": "stable",
                    "release_generation": "release-v7",
                    "membership_checksum": "c".repeat(64),
                    "cache_domains": ["cache-a", "cache-b"]
                }
            }
        })
        .to_string();
        let cohort_catalog = serde_json::json!({
            "version": "cohort-pair-v1",
            "pairs": {
                "p-v1_d-v1": {
                    "cache_domain": "cache-a",
                    "prefill_runtime_cohort": "p-v1",
                    "decode_runtime_cohort": "d-v1"
                }
            }
        })
        .to_string();
        let values = HashMap::from([
            ("SMG_PD_ENVIRONMENT", "test".to_string()),
            ("SMG_PD_SERVICE", "glm52".to_string()),
            ("SMG_PD_RELEASE", "stable".to_string()),
            (
                "SMG_PD_METRIC_CONTRACT_VERSION",
                "smg-pd-request-v1".to_string(),
            ),
            ("SMG_PD_SCHEMA_DIGEST", "a".repeat(64)),
            ("SMG_PD_IMAGE_DIGEST", format!("sha256:{}", "b".repeat(64))),
            (
                "SMG_PD_PRODUCER_REVISION",
                "0744ea180744ea180744ea180744ea180744ea18".to_string(),
            ),
            ("SMG_PD_REQUEST_CLASS_CATALOG_JSON", request_catalog),
            ("SMG_PD_COHORT_CATALOG_JSON", cohort_catalog),
        ]);
        Arc::new(
            PdLifecycleConfig::from_lookup(|key| values.get(key).cloned())
                .expect("valid catalog configuration")
                .expect("enabled contract"),
        )
    }

    fn with_test_recorder<T>(f: impl FnOnce() -> T) -> (String, T) {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let result = metrics::with_local_recorder(&recorder, f);
        (handle.render(), result)
    }

    fn has_evidence(rendered: &str, metric: &str, state: &str) -> bool {
        rendered.lines().any(|line| {
            line.starts_with("smg_pd_request_metric_evidence_total{")
                && line.contains(&format!(r#"metric="{metric}""#))
                && line.contains(&format!(r#"evidence_state="{state}""#))
        })
    }

    #[test]
    fn contract_anchor_exists_without_requests() {
        let (rendered, ()) = with_test_recorder(|| config().install_capability_anchor());

        assert!(rendered.contains("smg_pd_request_lifecycle_contract_info{"));
        assert!(rendered.contains(r#"pd_service="glm52""#));
        assert!(rendered.contains(r#"smg_release="stable""#));
        assert!(rendered.contains(r#"environment="test""#));
        assert!(rendered.contains(r#"metric_contract_version="smg-pd-request-v1""#));
        assert!(rendered.contains(r#"producer_transport="http""#));
        assert!(rendered.contains(r#"request_class_catalog_version="unassigned""#));
        assert!(rendered.contains(r#"cohort_catalog_version="unassigned""#));
        assert!(rendered.contains("} 1"));
    }

    #[test]
    fn every_lifecycle_family_is_environment_scoped_and_releases_do_not_merge() {
        let (rendered, ()) = with_test_recorder(|| {
            for environment in ["test", "prod"] {
                let config = config_for_environment(environment);
                config.install_capability_anchor();
                let request = PdRequestLifecycle::start(config, TrafficClass::Natural);
                request.begin_attempt(identity());
                request.observe_output_token_after(Duration::from_secs(1));
                request.observe_output_token_after(Duration::from_secs(2));
                request.finish(
                    PdTerminalOutcome::Success,
                    StatusCode::OK,
                    Some(PdUsage::new(8, 2)),
                );
            }
        });

        for family in [
            "smg_pd_request_lifecycle_contract_info",
            "smg_pd_requests_started_total",
            "smg_pd_terminal_requests_total",
            "smg_pd_request_metric_evidence_total",
            "smg_pd_e2e_seconds",
            "smg_pd_request_v1_ttft_seconds",
            "smg_pd_tpot_seconds",
            "smg_pd_completed_input_tokens_total",
            "smg_pd_completed_output_tokens_total",
        ] {
            let series = rendered
                .lines()
                .filter(|line| line.starts_with(family))
                .collect::<Vec<_>>();
            assert!(!series.is_empty(), "missing {family}");
            assert!(
                series.iter().all(|line| line.contains("environment=")),
                "{family} has a series without environment: {series:?}"
            );
            assert!(
                series
                    .iter()
                    .any(|line| line.contains(r#"environment="test""#)),
                "{family} is missing test"
            );
            assert!(
                series
                    .iter()
                    .any(|line| line.contains(r#"environment="prod""#)),
                "{family} is missing prod"
            );
        }

        let started = rendered
            .lines()
            .filter(|line| line.starts_with("smg_pd_requests_started_total{"))
            .collect::<Vec<_>>();
        assert_eq!(started.len(), 2);
        assert!(started.iter().all(|line| line.ends_with(" 1")));
    }

    #[test]
    fn classified_lifecycle_emits_catalogued_natural_bucket() {
        let config = config_with_catalogs();
        let principal = format!("auth:{}", "1".repeat(64));
        let tenant = RouteRequestMeta::new(TenantKey::new(principal));
        let classification = config
            .classify_public_http_request(&tenant, "/v1/chat/completions", None)
            .expect("catalogued natural principal");

        let (rendered, ()) = with_test_recorder(|| {
            let request = PdRequestLifecycle::start_classified(config, classification);
            request.begin_attempt(identity());
            request.finish(
                PdTerminalOutcome::Success,
                StatusCode::OK,
                Some(PdUsage::new(8, 1)),
            );
        });

        let terminal = rendered
            .lines()
            .find(|line| line.starts_with("smg_pd_terminal_requests_total{"))
            .expect("terminal series");
        assert!(terminal.contains(r#"traffic_class="natural""#));
        assert!(terminal.contains(r#"workload_bucket="agent-80k-v1""#));
    }

    #[test]
    fn started_and_terminal_are_exactly_once() {
        let (rendered, (first, second)) = with_test_recorder(|| {
            let request = PdRequestLifecycle::start(config(), TrafficClass::Natural);
            request.begin_attempt(identity());
            let first = request.finish(
                PdTerminalOutcome::Success,
                StatusCode::OK,
                Some(PdUsage::new(80, 3)),
            );
            let second = request.finish(PdTerminalOutcome::Error, StatusCode::BAD_GATEWAY, None);
            (first, second)
        });

        assert!(first);
        assert!(!second);
        assert!(rendered.contains("smg_pd_requests_started_total{"));
        assert!(rendered.contains("smg_pd_terminal_requests_total{"));
        assert_eq!(
            rendered.matches("smg_pd_terminal_requests_total{").count(),
            1
        );
    }

    #[test]
    fn dropping_unfinished_lifecycle_records_one_cancel_terminal() {
        let (rendered, ()) = with_test_recorder(|| {
            let request = PdRequestLifecycle::start(config(), TrafficClass::Natural);
            drop(request);
        });

        assert_eq!(
            rendered.matches("smg_pd_requests_started_total{").count(),
            1
        );
        assert_eq!(
            rendered.matches("smg_pd_terminal_requests_total{").count(),
            1
        );
        assert!(rendered.contains(r#"outcome="cancel""#));
        assert!(rendered.contains(r#"status_class="4xx""#));
    }

    #[test]
    fn lifecycle_ttft_uses_a_versioned_family_distinct_from_transport_ttft() {
        let (rendered, ()) = with_test_recorder(|| {
            let request = PdRequestLifecycle::start(config(), TrafficClass::Natural);
            request.begin_attempt(identity());
            request.observe_output_token_after(Duration::from_secs(1));
            request.finish(
                PdTerminalOutcome::Success,
                StatusCode::OK,
                Some(PdUsage::new(8, 1)),
            );
        });

        assert!(rendered.contains("smg_pd_request_v1_ttft_seconds_count{"));
        assert!(!rendered.contains("smg_pd_ttft_seconds_count{"));
    }

    #[test]
    fn retry_discards_prior_attempt_evidence_without_recounting_request() {
        let (rendered, ()) = with_test_recorder(|| {
            let request = PdRequestLifecycle::start(config(), TrafficClass::Natural);
            request.begin_attempt(
                PdExecutionIdentity::new("agent-80k-v1", "cache-a", "p-old", "d-old", "old-pair")
                    .expect("valid prior attempt"),
            );
            request.observe_json(&serde_json::json!({
                "text": "retryable partial",
                "meta_info": {"prompt_tokens": 800, "completion_tokens": 300}
            }));

            request.begin_attempt(identity());
            request.observe_json(&serde_json::json!({
                "text": "final",
                "meta_info": {"prompt_tokens": 8, "completion_tokens": 3}
            }));
            request.finish_stream(PdTerminalOutcome::Success, StatusCode::OK);
        });

        assert_eq!(
            rendered.matches("smg_pd_requests_started_total{").count(),
            1
        );
        assert_eq!(
            rendered.matches("smg_pd_terminal_requests_total{").count(),
            1
        );
        let input = rendered
            .lines()
            .find(|line| line.starts_with("smg_pd_completed_input_tokens_total{"))
            .expect("input token series");
        assert_eq!(input.split_whitespace().last(), Some("8"));
        assert!(!rendered.contains(r#"execution_cohort_pair_id="old-pair""#));
    }

    #[test]
    fn unknown_usage_does_not_emit_zero_token_counters() {
        let (rendered, ()) = with_test_recorder(|| {
            let request = PdRequestLifecycle::start(config(), TrafficClass::Natural);
            request.begin_attempt(identity());
            request.finish(PdTerminalOutcome::Error, StatusCode::BAD_GATEWAY, None);
        });

        assert!(has_evidence(&rendered, "usage", "unknown"));
        assert!(!rendered.contains("smg_pd_completed_input_tokens_total"));
        assert!(!rendered.contains("smg_pd_completed_output_tokens_total"));
    }

    #[test]
    fn tpot_uses_first_to_last_token_interval_and_n_minus_one() {
        let (rendered, ()) = with_test_recorder(|| {
            let request = PdRequestLifecycle::start(config(), TrafficClass::Natural);
            request.begin_attempt(identity());
            request.observe_output_token_after(Duration::from_secs(2));
            request.observe_output_token_after(Duration::from_secs(4));
            request.finish(
                PdTerminalOutcome::Success,
                StatusCode::OK,
                Some(PdUsage::new(80, 3)),
            );
        });

        assert!(has_evidence(&rendered, "tpot", "known"));
        let sum = rendered
            .lines()
            .find(|line| line.starts_with("smg_pd_tpot_seconds_sum{"))
            .expect("TPOT sum");
        assert!(
            sum.split_whitespace().last() == Some("1"),
            "(4s - 2s) / (3 - 1) = 1s: {rendered}"
        );
    }

    #[test]
    fn tpot_is_unknown_for_ineligible_requests() {
        for (traffic_class, outcome, output_tokens) in [
            (TrafficClass::Natural, PdTerminalOutcome::Success, 0),
            (TrafficClass::Natural, PdTerminalOutcome::Success, 1),
            (TrafficClass::Natural, PdTerminalOutcome::Cancel, 3),
            (TrafficClass::Synthetic, PdTerminalOutcome::Success, 3),
            (TrafficClass::Benchmark, PdTerminalOutcome::Success, 3),
        ] {
            let (rendered, ()) = with_test_recorder(|| {
                let request = PdRequestLifecycle::start(config(), traffic_class);
                request.begin_attempt(identity());
                request.observe_output_token_after(Duration::from_secs(1));
                request.observe_output_token_after(Duration::from_secs(2));
                request.finish(
                    outcome,
                    StatusCode::OK,
                    Some(PdUsage::new(4, output_tokens)),
                );
            });
            assert!(has_evidence(&rendered, "tpot", "unknown"));
            assert!(!rendered.contains("smg_pd_tpot_seconds_sum"));
        }
    }

    #[test]
    fn multi_choice_stream_fails_timing_evidence_closed() {
        let (rendered, ()) = with_test_recorder(|| {
            let request = PdRequestLifecycle::start(config(), TrafficClass::Natural);
            request.begin_attempt(identity());
            request.observe_json_after(
                &serde_json::json!({"choices": [{"index": 0, "delta": {"content": "a"}}]}),
                Duration::from_secs(1),
            );
            request.observe_json_after(
                &serde_json::json!({"choices": [{"index": 1, "delta": {"content": "b"}}]}),
                Duration::from_secs(2),
            );
            request.finish(
                PdTerminalOutcome::Success,
                StatusCode::OK,
                Some(PdUsage::new(8, 4)),
            );
        });

        assert!(has_evidence(&rendered, "ttft", "unknown"));
        assert!(has_evidence(&rendered, "tpot", "unknown"));
        assert!(!rendered.contains("smg_pd_request_v1_ttft_seconds_count{"));
        assert!(!rendered.contains("smg_pd_tpot_seconds_count{"));
    }

    #[test]
    fn choice_identity_tracking_is_bounded_after_timing_becomes_ambiguous() {
        let (rendered, tracked_choices) = with_test_recorder(|| {
            let request = PdRequestLifecycle::start(config(), TrafficClass::Natural);
            request.begin_attempt(identity());
            for index in 0..64 {
                request.observe_json_after(
                    &serde_json::json!({
                        "choices": [{"index": index, "delta": {"content": "x"}}]
                    }),
                    Duration::from_millis(index + 1),
                );
            }
            let tracked_choices = request
                .observation
                .lock()
                .unwrap_or_else(|lock| lock.into_inner())
                .choice_output_events
                .len();
            request.finish(
                PdTerminalOutcome::Success,
                StatusCode::OK,
                Some(PdUsage::new(8, 64)),
            );
            tracked_choices
        });

        assert_eq!(tracked_choices, 1);
        assert!(has_evidence(&rendered, "ttft", "unknown"));
        assert!(has_evidence(&rendered, "tpot", "unknown"));
    }

    #[test]
    fn nonzero_single_choice_index_is_observed_reliably() {
        let (rendered, ()) = with_test_recorder(|| {
            let request = PdRequestLifecycle::start(config(), TrafficClass::Natural);
            request.begin_attempt(identity());
            for (elapsed, content) in [(1, "a"), (2, "b")] {
                request.observe_json_after(
                    &serde_json::json!({
                        "choices": [{"index": 3, "delta": {"content": content}}]
                    }),
                    Duration::from_secs(elapsed),
                );
            }
            request.finish(
                PdTerminalOutcome::Success,
                StatusCode::OK,
                Some(PdUsage::new(8, 2)),
            );
        });

        assert!(has_evidence(&rendered, "ttft", "known"));
        assert!(has_evidence(&rendered, "tpot", "known"));
    }

    #[test]
    fn reasoning_content_contributes_to_ttft_and_tpot() {
        let (rendered, ()) = with_test_recorder(|| {
            let request = PdRequestLifecycle::start(config(), TrafficClass::Natural);
            request.begin_attempt(identity());
            for (elapsed, reasoning) in [(1, "think"), (3, " again")] {
                request.observe_json_after(
                    &serde_json::json!({
                        "choices": [{
                            "index": 0,
                            "delta": {"reasoning_content": reasoning}
                        }]
                    }),
                    Duration::from_secs(elapsed),
                );
            }
            request.finish(
                PdTerminalOutcome::Success,
                StatusCode::OK,
                Some(PdUsage::new(8, 3)),
            );
        });

        assert!(has_evidence(&rendered, "ttft", "known"));
        assert!(has_evidence(&rendered, "tpot", "known"));
        let ttft_sum = rendered
            .lines()
            .find(|line| line.starts_with("smg_pd_request_v1_ttft_seconds_sum{"))
            .expect("TTFT sum");
        assert_eq!(ttft_sum.split_whitespace().last(), Some("1"));
        let tpot_sum = rendered
            .lines()
            .find(|line| line.starts_with("smg_pd_tpot_seconds_sum{"))
            .expect("TPOT sum");
        assert_eq!(tpot_sum.split_whitespace().last(), Some("1"));
    }

    #[test]
    fn tpot_fails_closed_when_final_tokens_are_fewer_than_output_events() {
        let (rendered, ()) = with_test_recorder(|| {
            let request = PdRequestLifecycle::start(config(), TrafficClass::Natural);
            request.begin_attempt(identity());
            for elapsed in [1, 2, 3] {
                request.observe_json_after(
                    &serde_json::json!({
                        "choices": [{
                            "index": 0,
                            "delta": {"reasoning_content": "token"}
                        }]
                    }),
                    Duration::from_secs(elapsed),
                );
            }
            request.finish(
                PdTerminalOutcome::Success,
                StatusCode::OK,
                Some(PdUsage::new(8, 2)),
            );
        });

        assert!(has_evidence(&rendered, "ttft", "known"));
        assert!(has_evidence(&rendered, "tpot", "unknown"));
        assert!(!rendered.contains("smg_pd_tpot_seconds_count{"));
    }

    #[test]
    fn authenticated_natural_request_uses_server_catalog_bucket() {
        let config = config_with_catalogs();
        let principal = format!("auth:{}", "1".repeat(64));
        let tenant = RouteRequestMeta::new(TenantKey::new(principal));
        let mut forged = HeaderMap::new();
        forged.insert("x-smg-traffic-class", HeaderValue::from_static("benchmark"));
        forged.insert(
            "x-smg-workload-bucket",
            HeaderValue::from_static("attacker-chosen"),
        );
        forged.insert(
            PROBE_CACHE_DOMAIN_HEADER,
            HeaderValue::from_static("cache-b"),
        );

        let classified = config
            .classify_public_http_request(&tenant, "/v1/chat/completions", Some(&forged))
            .expect("client labels are ignored");
        assert_eq!(classified.traffic_class, TrafficClass::Natural);
        assert_eq!(&*classified.workload_bucket, "agent-80k-v1");

        let anonymous = RouteRequestMeta::new(TenantKey::from("anonymous"));
        let classified = config
            .classify_public_http_request(&anonymous, "/v1/chat/completions", Some(&forged))
            .expect("public cache-domain header is ignored");
        assert_eq!(classified.traffic_class, TrafficClass::Natural);
        assert_eq!(&*classified.workload_bucket, UNASSIGNED);
        assert!(classified.response_attribution.is_none());
    }

    #[test]
    fn privileged_class_requires_server_route_endpoint_and_authenticated_principal() {
        let config = config_with_catalogs();
        let principal = format!("auth:{}", "1".repeat(64));
        let tenant = RouteRequestMeta::new(TenantKey::new(principal));
        let mut headers = HeaderMap::new();
        headers.insert(
            PROBE_CACHE_DOMAIN_HEADER,
            HeaderValue::from_static("cache-a"),
        );
        let route_identity = config
            .authorize_trusted_http_route(
                &tenant,
                "internal-probe-v1",
                "/v1/chat/completions",
                &headers,
            )
            .expect("server route authorization");
        let tenant = tenant.with_extension(route_identity);

        let classified = config
            .classify_public_http_request(&tenant, "/v1/chat/completions", Some(&headers))
            .expect("exact trusted route tuple");
        assert_eq!(classified.traffic_class, TrafficClass::Synthetic);
        assert_eq!(&*classified.workload_bucket, "probe-2k-v1");
        let attribution = classified
            .response_attribution
            .as_ref()
            .expect("privileged route attribution");
        assert_eq!(attribution.environment(), "test");
        assert_eq!(attribution.traffic_class(), "synthetic");
        assert_eq!(attribution.pd_service(), "glm52");
        assert_eq!(attribution.smg_release(), "stable");
        assert_eq!(attribution.release_generation(), "release-v7");
        assert_eq!(attribution.membership_checksum(), "c".repeat(64));
        assert_eq!(attribution.cache_domain(), "cache-a");

        let response_headers = attribution
            .to_internal_probe_headers()
            .expect("validated attribution values are valid HTTP headers");
        assert_eq!(response_headers["x-smg-pd-service"], "glm52");
        assert_eq!(response_headers["x-smg-pd-release"], "stable");
        assert_eq!(response_headers["x-smg-pd-environment"], "test");
        assert_eq!(
            response_headers["x-smg-pd-release-generation"],
            "release-v7"
        );
        assert_eq!(
            response_headers["x-smg-pd-membership-checksum"],
            "c".repeat(64)
        );
        assert_eq!(response_headers["x-smg-pd-cache-domain"], "cache-a");
        assert_eq!(response_headers["x-smg-pd-traffic-class"], "synthetic");

        let mut public_response_headers = response_headers.clone();
        PdResponseAttribution::strip_internal_probe_headers(&mut public_response_headers);
        assert!(public_response_headers.is_empty());

        headers.insert(
            PROBE_CACHE_DOMAIN_HEADER,
            HeaderValue::from_static("cache-b"),
        );
        let cache_b = config
            .classify_public_http_request(&tenant, "/v1/chat/completions", Some(&headers))
            .expect("second catalogued cache domain");
        assert_eq!(
            cache_b
                .response_attribution
                .as_ref()
                .expect("cache-b attribution")
                .cache_domain(),
            "cache-b"
        );

        headers.insert(
            PROBE_CACHE_DOMAIN_HEADER,
            HeaderValue::from_static("cache-unknown"),
        );
        assert!(config
            .classify_public_http_request(&tenant, "/v1/chat/completions", Some(&headers))
            .is_err());
        assert!(config
            .classify_public_http_request(&tenant, "/v1/completions", Some(&headers))
            .is_err());

        let wrong_principal =
            RouteRequestMeta::new(TenantKey::new(format!("auth:{}", "2".repeat(64))))
                .with_extension(
                    PdTrustedRouteIdentity::new("internal-probe-v1").expect("valid route identity"),
                );
        assert!(config
            .classify_public_http_request(&wrong_principal, "/v1/chat/completions", None,)
            .is_err());
    }

    #[test]
    fn public_worker_identity_never_assigns_a_global_workload_bucket() {
        let prefill = HashMap::from([
            ("cache_domain".to_string(), "cache-a".to_string()),
            ("runtime_cohort".to_string(), "p-v1".to_string()),
            (
                "execution_cohort_pair_id".to_string(),
                "p-v1_d-v1".to_string(),
            ),
        ]);
        let decode = HashMap::from([
            ("runtime_cohort".to_string(), "d-v1".to_string()),
            (
                "execution_cohort_pair_id".to_string(),
                "p-v1_d-v1".to_string(),
            ),
        ]);

        let identity = config_with_catalogs().execution_identity_from_labels(&prefill, &decode);
        assert_eq!(&*identity.workload_bucket, "unassigned");
        assert_eq!(&*identity.cache_domain, "cache-a");
        assert_eq!(&*identity.prefill_runtime_cohort, "p-v1");
        assert_eq!(&*identity.decode_runtime_cohort, "d-v1");
    }

    #[test]
    fn mismatched_execution_pair_metadata_fails_closed() {
        let prefill = HashMap::from([
            ("cache_domain".to_string(), "cache-a".to_string()),
            ("runtime_cohort".to_string(), "p-v1".to_string()),
            ("execution_cohort_pair_id".to_string(), "pair-a".to_string()),
        ]);
        let decode = HashMap::from([
            ("runtime_cohort".to_string(), "d-v1".to_string()),
            ("execution_cohort_pair_id".to_string(), "pair-b".to_string()),
        ]);

        let identity = config_with_catalogs().execution_identity_from_labels(&prefill, &decode);
        assert_eq!(identity, PdExecutionIdentity::unassigned());
    }

    #[test]
    fn worker_labels_without_versioned_cohort_catalog_fail_closed() {
        let prefill = HashMap::from([
            ("cache_domain".to_string(), "cache-a".to_string()),
            ("runtime_cohort".to_string(), "p-v1".to_string()),
            (
                "execution_cohort_pair_id".to_string(),
                "p-v1_d-v1".to_string(),
            ),
        ]);
        let decode = HashMap::from([
            ("runtime_cohort".to_string(), "d-v1".to_string()),
            (
                "execution_cohort_pair_id".to_string(),
                "p-v1_d-v1".to_string(),
            ),
        ]);

        assert_eq!(
            config().execution_identity_from_labels(&prefill, &decode),
            PdExecutionIdentity::unassigned()
        );
    }

    #[test]
    fn environment_identity_is_all_or_nothing_and_validated() {
        let mut values = HashMap::from([
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
        ]);
        let parsed = PdLifecycleConfig::from_lookup(|key| values.get(key).map(ToString::to_string));
        assert!(parsed.expect("valid env").is_some());

        values.insert("SMG_PD_PRODUCER_REVISION", "0744ea18");
        assert!(
            PdLifecycleConfig::from_lookup(|key| values.get(key).map(ToString::to_string)).is_err()
        );
        values.insert(
            "SMG_PD_PRODUCER_REVISION",
            "0744ea180744ea180744ea180744ea180744ea18",
        );

        values.remove("SMG_PD_RELEASE");
        assert!(
            PdLifecycleConfig::from_lookup(|key| values.get(key).map(ToString::to_string)).is_err()
        );

        values.insert("SMG_PD_RELEASE", "bad release with spaces");
        assert!(
            PdLifecycleConfig::from_lookup(|key| values.get(key).map(ToString::to_string)).is_err()
        );

        values.insert("SMG_PD_RELEASE", "stable");
        values.insert("SMG_PD_ENVIRONMENT", "staging");
        assert!(
            PdLifecycleConfig::from_lookup(|key| values.get(key).map(ToString::to_string)).is_err()
        );

        values.remove("SMG_PD_ENVIRONMENT");
        assert!(
            PdLifecycleConfig::from_lookup(|key| values.get(key).map(ToString::to_string)).is_err()
        );
    }

    #[test]
    fn legacy_single_cache_domain_catalog_fails_closed() {
        let legacy = serde_json::json!({
            "version": "request-class-v1",
            "privileged_routes": {
                "internal-probe-v1": {
                    "authenticated_principal": format!("auth:{}", "1".repeat(64)),
                    "endpoint": "/v1/chat/completions",
                    "traffic_class": "synthetic",
                    "workload_bucket": "probe-2k-v1",
                    "pd_service": "glm52",
                    "smg_release": "stable",
                    "release_generation": "release-v7",
                    "membership_checksum": "c".repeat(64),
                    "cache_domain": "cache-a"
                }
            }
        })
        .to_string();

        assert!(parse_request_class_catalog(&legacy).is_err());
    }

    #[test]
    fn no_pd_environment_means_contract_is_disabled() {
        assert!(PdLifecycleConfig::from_lookup(|_| None)
            .expect("empty env is valid")
            .is_none());
    }

    #[test]
    fn empty_pd_environment_value_is_rejected() {
        assert!(PdLifecycleConfig::from_lookup(|key| {
            (key == "SMG_PD_SERVICE").then(String::new)
        })
        .is_err());
    }

    #[test]
    fn usage_requires_both_input_and_output() {
        assert_eq!(
            parse_usage(&serde_json::json!({
                "usage": {"prompt_tokens": 8, "completion_tokens": 3}
            })),
            Some(PdUsage::new(8, 3))
        );
        assert_eq!(
            parse_usage(&serde_json::json!({"usage": {"prompt_tokens": 8}})),
            None
        );
    }

    #[test]
    fn non_streaming_body_keeps_ttft_and_tpot_unknown() {
        let (rendered, ()) = with_test_recorder(|| {
            let request = PdRequestLifecycle::start(config(), TrafficClass::Natural);
            request.begin_attempt(identity());
            request.observe_non_stream_json(&serde_json::json!({
                "choices": [{"index": 0, "text": "complete body"}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 2}
            }));
            request.finish_stream(PdTerminalOutcome::Success, StatusCode::OK);
        });

        assert!(has_evidence(&rendered, "usage", "known"));
        assert!(has_evidence(&rendered, "ttft", "unknown"));
        assert!(has_evidence(&rendered, "tpot", "unknown"));
        assert!(!rendered.contains("smg_pd_request_v1_ttft_seconds_count{"));
    }

    #[test]
    fn sse_observer_handles_split_events_and_final_usage() {
        let request = PdRequestLifecycle::start(config(), TrafficClass::Natural);
        request.begin_attempt(identity());
        let mut observer = PdSseObserver::new(Arc::clone(&request));
        observer.observe_chunk(b"data: {\"text\":\"a\",\"meta_info\":{\"completion_tokens\":1}}\n");
        observer.observe_chunk(
            b"\ndata: {\"text\":\"b\",\"meta_info\":{\"prompt_tokens\":8,\"completion_tokens\":2}}\n\ndata: [DONE]\n\n",
        );

        let observation = request
            .observation
            .lock()
            .unwrap_or_else(|lock| lock.into_inner());
        assert_eq!(observation.output_events, 2);
        assert_eq!(observation.usage, Some(PdUsage::new(8, 2)));
    }

    #[test]
    fn predispatch_error_keeps_execution_identity_unassigned() {
        let (rendered, ()) = with_test_recorder(|| {
            let request = PdRequestLifecycle::start(config(), TrafficClass::Natural);
            request.finish(
                PdTerminalOutcome::Error,
                StatusCode::SERVICE_UNAVAILABLE,
                None,
            );
        });

        let terminal = rendered
            .lines()
            .find(|line| line.starts_with("smg_pd_terminal_requests_total{"))
            .expect("terminal series");
        assert!(terminal.contains(r#"cache_domain="unassigned""#));
        assert!(terminal.contains(r#"prefill_runtime_cohort="unassigned""#));
        assert!(terminal.contains(r#"decode_runtime_cohort="unassigned""#));
        assert!(terminal.contains(r#"execution_cohort_pair_id="unassigned""#));
    }

    #[test]
    fn partial_stream_usage_is_not_treated_as_final_usage() {
        let (rendered, ()) = with_test_recorder(|| {
            let request = PdRequestLifecycle::start(config(), TrafficClass::Natural);
            request.begin_attempt(identity());
            request.observe_json(&serde_json::json!({
                "text": "partial",
                "meta_info": {"prompt_tokens": 8, "completion_tokens": 1}
            }));
            request.finish_stream(PdTerminalOutcome::Error, StatusCode::OK);
        });

        assert!(has_evidence(&rendered, "usage", "unknown"));
        assert!(!rendered.contains("smg_pd_completed_input_tokens_total"));
        assert!(!rendered.contains("smg_pd_completed_output_tokens_total"));
    }

    #[test]
    fn dropping_stream_guard_records_one_cancel_terminal() {
        let (rendered, second_finish) = with_test_recorder(|| {
            let request = PdRequestLifecycle::start(config(), TrafficClass::Natural);
            request.begin_attempt(identity());
            drop(PdStreamCancellationGuard::new(
                Some(Arc::clone(&request)),
                StatusCode::OK,
            ));
            request.finish_stream(PdTerminalOutcome::Error, StatusCode::BAD_GATEWAY)
        });

        assert!(!second_finish);
        assert_eq!(
            rendered.matches("smg_pd_terminal_requests_total{").count(),
            1
        );
        assert!(rendered.contains(r#"outcome="cancel""#));
    }

    #[test]
    fn mixed_execution_cohorts_are_not_merged() {
        let (rendered, ()) = with_test_recorder(|| {
            for identity in [
                PdExecutionIdentity::new("agent-80k-v1", "cache-a", "p-v1", "d-v1", "p-v1_d-v1")
                    .expect("valid pair"),
                PdExecutionIdentity::new("agent-80k-v1", "cache-a", "p-v2", "d-v1", "p-v2_d-v1")
                    .expect("valid pair"),
            ] {
                let request = PdRequestLifecycle::start(config(), TrafficClass::Natural);
                request.begin_attempt(identity);
                request.finish(
                    PdTerminalOutcome::Success,
                    StatusCode::OK,
                    Some(PdUsage::new(8, 2)),
                );
            }
        });

        let terminals = rendered
            .lines()
            .filter(|line| line.starts_with("smg_pd_terminal_requests_total{"))
            .collect::<Vec<_>>();
        assert_eq!(terminals.len(), 2);
        assert!(terminals
            .iter()
            .any(|line| line.contains(r#"execution_cohort_pair_id="p-v1_d-v1""#)));
        assert!(terminals
            .iter()
            .any(|line| line.contains(r#"execution_cohort_pair_id="p-v2_d-v1""#)));
    }

    #[test]
    fn known_zero_output_usage_is_distinct_from_unknown_usage() {
        let (rendered, ()) = with_test_recorder(|| {
            let request = PdRequestLifecycle::start(config(), TrafficClass::Natural);
            request.begin_attempt(identity());
            request.finish(
                PdTerminalOutcome::Success,
                StatusCode::OK,
                Some(PdUsage::new(8, 0)),
            );
        });

        assert!(has_evidence(&rendered, "usage", "known"));
        assert!(rendered.contains("smg_pd_completed_input_tokens_total{"));
        assert!(rendered.contains("smg_pd_completed_output_tokens_total{"));
    }
}
