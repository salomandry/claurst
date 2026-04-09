// registry.rs — Registry of all available LLM providers.
//
// Holds an `Arc<dyn LlmProvider>` for each registered provider and exposes
// lookup, health-check, and default-provider helpers.

use std::collections::HashMap;
use std::sync::Arc;

use claurst_core::ProviderId;

use crate::client::ClientConfig;
use crate::provider::LlmProvider;
use crate::provider_types::ProviderStatus;
use crate::providers::{
    normalize_openai_base_url, AnthropicProvider, AzureProvider, BedrockProvider, CodexProvider,
    CohereProvider, CopilotProvider, GoogleProvider, MinimaxProvider, OpenAiProvider,
};

/// Registry of all available LLM providers.
/// Holds `Arc<dyn LlmProvider>` for each registered provider.
pub struct ProviderRegistry {
    providers: HashMap<ProviderId, Arc<dyn LlmProvider>>,
    default_provider_id: ProviderId,
}

fn provider_from_key(provider_id: &str, key: String) -> Option<Arc<dyn LlmProvider>> {
    use crate::providers::openai_compat_providers as p;

    match provider_id {
        "anthropic" => Some(Arc::new(AnthropicProvider::from_config(
            ClientConfig { api_key: key, ..Default::default() },
        ))),
        "minimax" => Some(Arc::new(MinimaxProvider::new(key))),
        "openai" => Some(Arc::new(OpenAiProvider::new(key))),
        "google" => Some(Arc::new(GoogleProvider::new(key))),
        "github-copilot" => Some(Arc::new(CopilotProvider::new(key))),
        "codex" => {
            // The Codex provider is OAuth-based; the `key` field is not used.
            // Load from the stored token file instead.
            CodexProvider::from_stored().map(|p| Arc::new(p) as Arc<dyn LlmProvider>)
        }
        "cohere" => Some(Arc::new(CohereProvider::new(key))),
        "groq" => Some(Arc::new(p::groq().with_api_key(key))),
        "mistral" => Some(Arc::new(p::mistral().with_api_key(key))),
        "deepseek" => Some(Arc::new(p::deepseek().with_api_key(key))),
        "xai" => Some(Arc::new(p::xai().with_api_key(key))),
        "openrouter" => Some(Arc::new(p::openrouter().with_api_key(key))),
        "togetherai" | "together-ai" => Some(Arc::new(p::together_ai().with_api_key(key))),
        "perplexity" => Some(Arc::new(p::perplexity().with_api_key(key))),
        "cerebras" => Some(Arc::new(p::cerebras().with_api_key(key))),
        "deepinfra" => Some(Arc::new(p::deepinfra().with_api_key(key))),
        "venice" => Some(Arc::new(p::venice().with_api_key(key))),
        "huggingface" => Some(Arc::new(p::huggingface().with_api_key(key))),
        "nvidia" => Some(Arc::new(p::nvidia().with_api_key(key))),
        "siliconflow" => Some(Arc::new(p::siliconflow().with_api_key(key))),
        "sambanova" => Some(Arc::new(p::sambanova().with_api_key(key))),
        "moonshot" => Some(Arc::new(p::moonshot().with_api_key(key))),
        "zhipu" => Some(Arc::new(p::zhipu().with_api_key(key))),
        "qwen" => Some(Arc::new(p::qwen().with_api_key(key))),
        "nebius" => Some(Arc::new(p::nebius().with_api_key(key))),
        "novita" => Some(Arc::new(p::novita().with_api_key(key))),
        "ovhcloud" => Some(Arc::new(p::ovhcloud().with_api_key(key))),
        "scaleway" => Some(Arc::new(p::scaleway().with_api_key(key))),
        "vultr" | "vultr-ai" => Some(Arc::new(p::vultr_ai().with_api_key(key))),
        "baseten" => Some(Arc::new(p::baseten().with_api_key(key))),
        "friendli" => Some(Arc::new(p::friendli().with_api_key(key))),
        "upstage" => Some(Arc::new(p::upstage().with_api_key(key))),
        "stepfun" => Some(Arc::new(p::stepfun().with_api_key(key))),
        "fireworks" => Some(Arc::new(p::fireworks().with_api_key(key))),
        _ => None,
    }
}

/// API key for a provider: `config.provider_configs.<id>.api_key` from settings.json
/// wins, then [`claurst_core::AuthStore::api_key_for`] (auth.json + env vars).
fn config_aware_api_key(provider_id: &str, config: &claurst_core::config::Config) -> Option<String> {
    if let Some(pc) = config.provider_configs.get(provider_id) {
        if let Some(ref k) = pc.api_key {
            if !k.trim().is_empty() {
                return Some(k.clone());
            }
        }
    }
    claurst_core::AuthStore::load().api_key_for(provider_id)
}

fn runtime_provider_for_with_config(
    provider_id: &str,
    config: &claurst_core::config::Config,
) -> Option<Arc<dyn LlmProvider>> {
    use crate::providers::openai_compat_providers as p;

    // Local providers never require an API key — build them directly so that
    // the auth-store bypass below doesn't silently drop them.
    // Accept both the hyphenated canonical IDs ("llama-cpp", "lm-studio") and
    // the non-hyphenated aliases ("llamacpp", "lmstudio") used throughout the
    // TUI / connect dialog.
    match provider_id {
        "ollama" => return Some(Arc::new(p::ollama())),
        "lmstudio" | "lm-studio" => return Some(Arc::new(p::lm_studio())),
        // "llama-server" is the binary name for the modern llama.cpp server.
        "llamacpp" | "llama-cpp" | "llama-server" => return Some(Arc::new(p::llama_cpp())),
        _ => {}
    }

    let key = config_aware_api_key(provider_id, config)?;
    if key.is_empty() {
        return None;
    }
    provider_from_key(provider_id, key)
}

/// Build a provider from auth store / env only (no `settings.json` per-provider keys).
/// Prefer [`provider_with_config_overrides`] when a [`Config`] is available.
pub fn runtime_provider_for(provider_id: &str) -> Option<Arc<dyn LlmProvider>> {
    runtime_provider_for_with_config(provider_id, &claurst_core::config::Config::default())
}

/// Resolve a provider the same way as the query coordinator: prefer a fresh
/// provider using `provider_configs.<id>.api_key` from settings when set, then
/// auth store + env (so e.g. `OPENAI_BASE_URL` works with a key only in
/// `settings.json`), fall back to the pre-registered registry entry, then apply
/// `provider_configs` `api_base` overrides for OpenAI and local OpenAI-compat servers.
///
/// Use this for **model listing** and any UI path that previously only called
/// [`ProviderRegistry::get`], so custom base URLs match chat requests.
pub fn provider_with_config_overrides(
    registry: &ProviderRegistry,
    provider_id: &str,
    config: &claurst_core::config::Config,
) -> Option<Arc<dyn LlmProvider>> {
    use crate::providers::openai_compat_providers as p;

    let pid = ProviderId::new(provider_id);
    let runtime_provider = runtime_provider_for_with_config(provider_id, config);
    let registry_provider = if runtime_provider.is_some() {
        None
    } else {
        registry.get(&pid).cloned()
    };
    let mut provider = runtime_provider.or(registry_provider)?;

    if let Some(override_base) = config
        .provider_configs
        .get(provider_id)
        .and_then(|pc| pc.api_base.as_deref())
    {
        let trimmed = override_base.trim_end_matches('/');
        let base_url = if trimmed.ends_with("/v1") {
            trimmed.to_string()
        } else {
            format!("{}/v1", trimmed)
        };
        let overridden: Option<Arc<dyn LlmProvider>> = match provider_id {
            "openai" => config_aware_api_key("openai", config).and_then(|key| {
                normalize_openai_base_url(override_base).map(|base| {
                    Arc::new(OpenAiProvider::new(key).with_base_url(base)) as Arc<dyn LlmProvider>
                })
            }),
            "ollama" => Some(Arc::new(p::ollama().with_base_url(base_url))),
            "lmstudio" | "lm-studio" => Some(Arc::new(p::lm_studio().with_base_url(base_url))),
            "llamacpp" | "llama-cpp" | "llama-server" => {
                Some(Arc::new(p::llama_cpp().with_base_url(base_url)))
            }
            _ => None,
        };
        if let Some(o) = overridden {
            provider = o;
        }
    }
    Some(provider)
}

impl ProviderRegistry {
    /// Create an empty registry with Anthropic as the default provider ID.
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
            default_provider_id: ProviderId::new(ProviderId::ANTHROPIC),
        }
    }

    /// Register a provider. Returns `&mut self` for builder chaining.
    pub fn register(&mut self, provider: Arc<dyn LlmProvider>) -> &mut Self {
        let id = provider.id().clone();
        self.providers.insert(id, provider);
        self
    }

    /// Set the default provider by ID.
    ///
    /// # Panics
    /// Panics if no provider with that ID has been registered.
    pub fn set_default(&mut self, id: ProviderId) -> &mut Self {
        assert!(
            self.providers.contains_key(&id),
            "set_default: provider '{}' is not registered",
            id,
        );
        self.default_provider_id = id;
        self
    }

    /// Get a provider by ID.
    pub fn get(&self, id: &ProviderId) -> Option<&Arc<dyn LlmProvider>> {
        self.providers.get(id)
    }

    /// Get the default provider.
    pub fn default_provider(&self) -> Option<&Arc<dyn LlmProvider>> {
        self.providers.get(&self.default_provider_id)
    }

    /// Get the default provider ID.
    pub fn default_provider_id(&self) -> &ProviderId {
        &self.default_provider_id
    }

    /// List all registered provider IDs.
    pub fn provider_ids(&self) -> Vec<&ProviderId> {
        self.providers.keys().collect()
    }

    /// Check health of all providers sequentially.
    /// Returns `(provider_id, status)` pairs.
    pub async fn check_all_health(&self) -> Vec<(ProviderId, ProviderStatus)> {
        let mut results = Vec::new();
        for (id, provider) in &self.providers {
            let status = provider
                .health_check()
                .await
                .unwrap_or(ProviderStatus::Unavailable {
                    reason: "health check failed".to_string(),
                });
            results.push((id.clone(), status));
        }
        results
    }

    /// Convenience: build a registry with just Anthropic registered as the
    /// default provider.  Takes the same [`ClientConfig`] that
    /// [`AnthropicClient`] takes.
    ///
    /// [`AnthropicClient`]: crate::client::AnthropicClient
    pub fn with_anthropic(config: ClientConfig) -> Self {
        let mut registry = Self::new();
        let provider = Arc::new(AnthropicProvider::from_config(config));
        registry.register(provider);
        registry
    }

    /// Register [`GoogleProvider`] if `GOOGLE_API_KEY` or
    /// `GOOGLE_GENERATIVE_AI_API_KEY` is set in the environment.
    /// Returns `&mut self` for builder chaining.
    pub fn with_google_if_key_set(&mut self) -> &mut Self {
        let key = std::env::var("GOOGLE_API_KEY")
            .or_else(|_| std::env::var("GOOGLE_GENERATIVE_AI_API_KEY"));
        if let Ok(key) = key {
            let provider = Arc::new(GoogleProvider::new(key));
            self.register(provider);
        }
        self
    }

    /// Register [`OpenAiProvider`] if `OPENAI_API_KEY` is set in the
    /// environment.  The API origin is `OPENAI_BASE_URL` when set, else
    /// `https://api.openai.com`.  Returns `&mut self` for builder chaining.
    pub fn with_openai_if_key_set(&mut self) -> &mut Self {
        if let Ok(key) = std::env::var("OPENAI_API_KEY") {
            let provider = Arc::new(OpenAiProvider::new(key));
            self.register(provider);
        }
        self
    }

    /// Register [`AzureProvider`] if `AZURE_API_KEY` and `AZURE_RESOURCE_NAME`
    /// are set in the environment.  Returns `&mut self` for builder chaining.
    pub fn with_azure_if_configured(&mut self) -> &mut Self {
        if let Some(p) = AzureProvider::from_env() {
            self.register(Arc::new(p));
        }
        self
    }

    /// Register [`BedrockProvider`] if AWS credentials are available in the
    /// environment (`AWS_ACCESS_KEY_ID`+`AWS_SECRET_ACCESS_KEY` or
    /// `AWS_BEARER_TOKEN_BEDROCK`).  Returns `&mut self` for builder chaining.
    pub fn with_bedrock_if_configured(&mut self) -> &mut Self {
        if let Some(p) = BedrockProvider::from_env() {
            self.register(Arc::new(p));
        }
        self
    }

    /// Register [`CopilotProvider`] if `GITHUB_TOKEN` is set in the environment.
    /// Returns `&mut self` for builder chaining.
    pub fn with_copilot_if_configured(&mut self) -> &mut Self {
        if let Some(p) = CopilotProvider::from_env() {
            self.register(Arc::new(p));
        }
        self
    }

    /// Register [`CodexProvider`] if stored Codex OAuth tokens are available in
    /// `~/.claurst/codex_tokens.json`.  Returns `&mut self` for builder chaining.
    pub fn with_codex_if_configured(&mut self) -> &mut Self {
        if let Some(p) = CodexProvider::from_stored() {
            self.register(Arc::new(p));
        }
        self
    }

    /// Register [`CohereProvider`] if `COHERE_API_KEY` is set in the environment.
    /// Returns `&mut self` for builder chaining.
    pub fn with_cohere_if_key_set(&mut self) -> &mut Self {
        if let Some(p) = CohereProvider::from_env() {
            self.register(Arc::new(p));
        }
        self
    }

    /// Build a registry with **all** providers that have credentials configured
    /// in the environment.  Anthropic is always the default provider.
    ///
    /// This is the recommended constructor for production use.
    pub fn from_environment(anthropic_config: ClientConfig) -> Self {
        let mut registry = Self::with_anthropic(anthropic_config);
        registry
            .with_openai_if_key_set()
            .with_google_if_key_set()
            .with_azure_if_configured()
            .with_bedrock_if_configured()
            .with_copilot_if_configured()
            .with_codex_if_configured()
            .with_cohere_if_key_set()
            .with_available_providers();
        registry
    }

    /// Build a registry that checks **both** environment variables and the
    /// persistent [`AuthStore`] (`~/.claurst/auth.json`) for credentials.
    ///
    /// This ensures that API keys stored via `/connect` or `claurst auth` are
    /// picked up at startup, not just env vars.  Falls back to
    /// `from_environment` for providers that only support env-var config, and
    /// adds any extra providers that have keys in the auth store.
    ///
    /// [`AuthStore`]: claurst_core::AuthStore
    pub fn from_environment_with_auth_store(anthropic_config: ClientConfig) -> Self {
        // Start with env-based registration.
        let mut registry = Self::from_environment(anthropic_config);

        // Now check the auth store for providers that weren't registered from
        // env vars.
        let auth_store = claurst_core::AuthStore::load();

        for (provider_id, _cred) in &auth_store.credentials {
            let pid = claurst_core::ProviderId::new(provider_id.as_str());
            // Skip if already registered from env vars.
            if registry.get(&pid).is_some() {
                continue;
            }
            // Try to get a usable key from the auth store.
            if let Some(key) = auth_store.api_key_for(provider_id) {
                if key.is_empty() {
                    continue;
                }
                let provider = provider_from_key(provider_id, key);
                if let Some(p) = provider {
                    registry.register(p);
                }
            }
        }

        registry
    }

    /// Register all providers that have environment variable credentials set.
    ///
    /// Local providers (Ollama, LM Studio, llama.cpp) are always registered
    /// regardless of credentials — `health_check()` will report them as
    /// unavailable if the server is not running.
    ///
    /// Remote API-key providers are only registered when their respective
    /// environment variables are set (non-empty).
    ///
    /// Returns `&mut self` for builder chaining.
    pub fn with_available_providers(&mut self) -> &mut Self {
        use crate::providers::openai_compat_providers as p;

        // Local providers — always try to register.
        self.register(Arc::new(p::ollama()));
        self.register(Arc::new(p::lm_studio()));
        self.register(Arc::new(p::llama_cpp()));

        // Remote providers — only register when an API key is present.
        if std::env::var("DEEPSEEK_API_KEY").map(|v| !v.is_empty()).unwrap_or(false) {
            self.register(Arc::new(p::deepseek()));
        }
        if std::env::var("GROQ_API_KEY").map(|v| !v.is_empty()).unwrap_or(false) {
            self.register(Arc::new(p::groq()));
        }
        if std::env::var("XAI_API_KEY").map(|v| !v.is_empty()).unwrap_or(false) {
            self.register(Arc::new(p::xai()));
        }
        if std::env::var("OPENROUTER_API_KEY").map(|v| !v.is_empty()).unwrap_or(false) {
            self.register(Arc::new(p::openrouter()));
        }
        if std::env::var("TOGETHER_API_KEY").map(|v| !v.is_empty()).unwrap_or(false) {
            self.register(Arc::new(p::together_ai()));
        }
        if std::env::var("PERPLEXITY_API_KEY").map(|v| !v.is_empty()).unwrap_or(false) {
            self.register(Arc::new(p::perplexity()));
        }
        if std::env::var("CEREBRAS_API_KEY").map(|v| !v.is_empty()).unwrap_or(false) {
            self.register(Arc::new(p::cerebras()));
        }
        if std::env::var("DEEPINFRA_API_KEY").map(|v| !v.is_empty()).unwrap_or(false) {
            self.register(Arc::new(p::deepinfra()));
        }
        if std::env::var("VENICE_API_KEY").map(|v| !v.is_empty()).unwrap_or(false) {
            self.register(Arc::new(p::venice()));
        }
        if std::env::var("DASHSCOPE_API_KEY").map(|v| !v.is_empty()).unwrap_or(false) {
            self.register(Arc::new(p::qwen()));
        }
        if std::env::var("MISTRAL_API_KEY").map(|v| !v.is_empty()).unwrap_or(false) {
            self.register(Arc::new(p::mistral()));
        }
        if std::env::var("SAMBANOVA_API_KEY").map(|v| !v.is_empty()).unwrap_or(false) {
            self.register(Arc::new(p::sambanova()));
        }
        if std::env::var("HF_TOKEN").map(|v| !v.is_empty()).unwrap_or(false) {
            self.register(Arc::new(p::huggingface()));
        }
        if std::env::var("MINIMAX_API_KEY").map(|v| !v.is_empty()).unwrap_or(false) {
            let key = std::env::var("MINIMAX_API_KEY").unwrap_or_default();
            self.register(Arc::new(MinimaxProvider::new(key)));
        }
        if std::env::var("NVIDIA_API_KEY").map(|v| !v.is_empty()).unwrap_or(false) {
            self.register(Arc::new(p::nvidia()));
        }
        if std::env::var("SILICONFLOW_API_KEY").map(|v| !v.is_empty()).unwrap_or(false) {
            self.register(Arc::new(p::siliconflow()));
        }
        if std::env::var("MOONSHOT_API_KEY").map(|v| !v.is_empty()).unwrap_or(false) {
            self.register(Arc::new(p::moonshot()));
        }
        if std::env::var("ZHIPU_API_KEY").map(|v| !v.is_empty()).unwrap_or(false) {
            self.register(Arc::new(p::zhipu()));
        }
        if std::env::var("NEBIUS_API_KEY").map(|v| !v.is_empty()).unwrap_or(false) {
            self.register(Arc::new(p::nebius()));
        }
        if std::env::var("NOVITA_API_KEY").map(|v| !v.is_empty()).unwrap_or(false) {
            self.register(Arc::new(p::novita()));
        }
        if std::env::var("OVHCLOUD_API_KEY").map(|v| !v.is_empty()).unwrap_or(false) {
            self.register(Arc::new(p::ovhcloud()));
        }
        if std::env::var("SCALEWAY_API_KEY").map(|v| !v.is_empty()).unwrap_or(false) {
            self.register(Arc::new(p::scaleway()));
        }
        if std::env::var("VULTR_API_KEY").map(|v| !v.is_empty()).unwrap_or(false) {
            self.register(Arc::new(p::vultr_ai()));
        }
        if std::env::var("BASETEN_API_KEY").map(|v| !v.is_empty()).unwrap_or(false) {
            self.register(Arc::new(p::baseten()));
        }
        if std::env::var("FRIENDLI_TOKEN").map(|v| !v.is_empty()).unwrap_or(false) {
            self.register(Arc::new(p::friendli()));
        }
        if std::env::var("UPSTAGE_API_KEY").map(|v| !v.is_empty()).unwrap_or(false) {
            self.register(Arc::new(p::upstage()));
        }
        if std::env::var("STEPFUN_API_KEY").map(|v| !v.is_empty()).unwrap_or(false) {
            self.register(Arc::new(p::stepfun()));
        }
        if std::env::var("FIREWORKS_API_KEY").map(|v| !v.is_empty()).unwrap_or(false) {
            self.register(Arc::new(p::fireworks()));
        }
        self
    }
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self::new()
    }
}
