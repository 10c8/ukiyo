mod openai;

pub struct GenOptions {
    pub prompt: String,
    pub context: Option<String>,
    pub max_tokens: u32,
    pub temperature: f32,
    pub stop: Vec<String>,
}

pub enum LLMType {
    OpenAI {
        model: &'static str,
        api_key: &'static str,
        org: &'static str,
    },
    #[allow(dead_code)]
    OpenAICompat {
        address: &'static str,
        port: &'static str,
        model: &'static str,
    },
}

pub struct LLMConfig {
    pub llm_type: LLMType,
    pub max_tokens: u32,
    pub temperature: f32,
}

impl Default for LLMConfig {
    fn default() -> Self {
        LLMConfig {
            llm_type: LLMType::OpenAI {
                model: "",
                api_key: "",
                org: "",
            },
            max_tokens: 64,
            temperature: 0.7,
        }
    }
}

pub struct LLM {
    config: LLMConfig,
}

impl LLM {
    pub fn new(config: LLMConfig) -> Self {
        LLM { config }
    }

    pub fn generate(
        &self,
        prompt: &str,
        context: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        stop: Option<Vec<String>>,
    ) -> String {
        let options = GenOptions {
            prompt: prompt.to_string(),
            context,
            max_tokens: max_tokens.unwrap_or(self.config.max_tokens),
            temperature: temperature.unwrap_or(self.config.temperature),
            stop: stop.unwrap_or(vec!["\n".to_string()]),
        };

        match self.config.llm_type {
            LLMType::OpenAI {
                model,
                api_key,
                org,
            } => openai::generate(options, model, api_key, org),
            LLMType::OpenAICompat { .. } => {
                todo!()
            }
        }
    }
}
