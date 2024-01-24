use reqwest::header::HeaderMap;
use serde::{Deserialize, Serialize};

use super::GenOptions;

#[derive(Debug, Serialize, Deserialize)]
struct CompletionsMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct CompletionsRequest {
    model: String,
    messages: Vec<CompletionsMessage>,
    temperature: f32,
    stop: Vec<String>,
}

#[derive(Serialize, Deserialize)]
struct CompletionsResponseUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
    models: usize,
}

#[derive(Serialize, Deserialize)]
struct CompletionsResponseChoice {
    message: CompletionsMessage,
    logprobs: Option<()>,
    finish_reason: String,
    index: usize,
}

#[derive(Serialize, Deserialize)]
struct CompletionsResponse {
    id: String,
    object: String,
    created: usize,
    model: String,
    choices: Vec<CompletionsResponseChoice>,
}

pub fn generate(options: GenOptions, model: &str, api_key: &str, org: &str) -> String {
    // let url = "https://api.openai.com/v1/chat/completions";
    let url = "http://127.0.0.1:8080/v1/chat/completions";

    let mut headers = HeaderMap::new();
    headers.insert("Content-Type", "application/json".parse().unwrap());
    headers.insert(
        "Authorization",
        format!("Bearer {}", api_key).parse().unwrap(),
    );
    headers.insert("OpenAI-Organization", org.parse().unwrap());

    let mut messages = Vec::new();

    if options.context.is_some() {
        messages.push(CompletionsMessage {
            role: "system".to_string(),
            content: format!("<s>[INST] {} [/INST]", options.context.unwrap()),
        });
    }

    messages.push(CompletionsMessage {
        role: "user".to_string(),
        content: format!("[INST] {} [/INST]", options.prompt),
    });

    let body = CompletionsRequest {
        model: model.to_string(),
        messages,
        temperature: options.temperature,
        stop: options.stop,
    };
    let body = serde_json::to_string(&body).unwrap();

    let client = reqwest::blocking::Client::new();
    let res = client.post(url).headers(headers).body(body).send().unwrap();
    let res = res.json().unwrap();

    // println!("{:#?}", res);

    let res: CompletionsResponse = serde_json::from_value(res).unwrap();
    let res = res.choices[0].message.content.clone();
    res
}
