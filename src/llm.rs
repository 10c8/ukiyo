use ecow::EcoString;

mod openai;

pub struct GenOptions {
    pub prompt: EcoString,
    pub context: Option<EcoString>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub stop: Vec<EcoString>,
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
    pub max_tokens: usize,
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
            max_tokens: 512,
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
        context: Option<EcoString>,
        max_tokens: Option<usize>,
        temperature: Option<f32>,
        stop: Option<Vec<EcoString>>,
    ) -> String {
        let options = GenOptions {
            prompt: prompt.into(),
            context,
            max_tokens: max_tokens.unwrap_or(self.config.max_tokens),
            temperature: temperature.unwrap_or(self.config.temperature),
            stop: stop.unwrap_or(vec!["\n".into()]),
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
