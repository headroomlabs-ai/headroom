use std::collections::BTreeMap;

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderPath {
    Health,
    AnthropicMessages,
    OpenAiChatCompletions,
    OpenAiResponses,
    OpenAiConversations,
    OpenAiConversation,
    OpenAiConversationItems,
    OpenAiConversationItem,
    BedrockInvoke,
    BedrockConverse,
    BedrockInvokeStream,
    BedrockConverseStream,
    VertexRawPredict,
    VertexStreamRawPredict,
    Generic,
}

impl ProviderPath {
    pub fn classify(path: &str) -> Self {
        if path == "/healthz" {
            return Self::Health;
        }
        if path == "/v1/messages" {
            return Self::AnthropicMessages;
        }
        if path == "/v1/chat/completions" {
            return Self::OpenAiChatCompletions;
        }
        if path == "/v1/responses" {
            return Self::OpenAiResponses;
        }
        if path == "/v1/conversations" {
            return Self::OpenAiConversations;
        }
        if is_conversation_item(path) {
            return Self::OpenAiConversationItem;
        }
        if is_conversation_items(path) {
            return Self::OpenAiConversationItems;
        }
        if is_conversation(path) {
            return Self::OpenAiConversation;
        }
        if path.starts_with("/model/") {
            if path.ends_with("/invoke-with-response-stream") {
                return Self::BedrockInvokeStream;
            }
            if path.ends_with("/converse-stream") {
                return Self::BedrockConverseStream;
            }
            if path.ends_with("/invoke") {
                return Self::BedrockInvoke;
            }
            if path.ends_with("/converse") {
                return Self::BedrockConverse;
            }
        }
        if path.starts_with("/v1beta1/projects/")
            && path.contains("/publishers/anthropic/models/")
            && path.ends_with(":rawPredict")
        {
            return Self::VertexRawPredict;
        }
        if path.starts_with("/v1beta1/projects/")
            && path.contains("/publishers/anthropic/models/")
            && path.ends_with(":streamRawPredict")
        {
            return Self::VertexStreamRawPredict;
        }
        Self::Generic
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Health => "health",
            Self::AnthropicMessages => "anthropic.messages",
            Self::OpenAiChatCompletions => "openai.chat_completions",
            Self::OpenAiResponses => "openai.responses",
            Self::OpenAiConversations => "openai.conversations",
            Self::OpenAiConversation => "openai.conversation",
            Self::OpenAiConversationItems => "openai.conversation_items",
            Self::OpenAiConversationItem => "openai.conversation_item",
            Self::BedrockInvoke => "bedrock.invoke",
            Self::BedrockConverse => "bedrock.converse",
            Self::BedrockInvokeStream => "bedrock.invoke_stream",
            Self::BedrockConverseStream => "bedrock.converse_stream",
            Self::VertexRawPredict => "vertex.raw_predict",
            Self::VertexStreamRawPredict => "vertex.stream_raw_predict",
            Self::Generic => "generic",
        }
    }
}

#[derive(Debug, Clone)]
pub struct SimulatedResponse {
    pub status: u16,
    pub content_type: &'static str,
    pub headers: BTreeMap<String, String>,
    pub body: Bytes,
}

impl SimulatedResponse {
    pub fn json(status: u16, value: Value) -> Self {
        let body = serde_json::to_vec(&value).expect("static simulator JSON serializes");
        Self {
            status,
            content_type: "application/json",
            headers: BTreeMap::new(),
            body: Bytes::from(body),
        }
    }

    pub fn text(status: u16, content_type: &'static str, body: impl Into<Bytes>) -> Self {
        Self {
            status,
            content_type,
            headers: BTreeMap::new(),
            body: body.into(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RequestFacts {
    pub method: String,
    pub path: String,
    pub provider_path: ProviderPath,
    pub body: Bytes,
    pub json: Option<Value>,
    pub wants_stream: bool,
    pub wants_bedrock_eventstream: bool,
}

impl RequestFacts {
    pub fn new(method: &str, path: &str, headers: &http::HeaderMap, body: Bytes) -> Self {
        let json = serde_json::from_slice::<Value>(&body).ok();
        let wants_stream = json
            .as_ref()
            .and_then(|v| v.get("stream"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
            || header_contains(headers, http::header::ACCEPT.as_str(), "text/event-stream");
        let wants_bedrock_eventstream = header_contains(
            headers,
            http::header::ACCEPT.as_str(),
            "application/vnd.amazon.eventstream",
        );
        Self {
            method: method.to_ascii_uppercase(),
            path: path.to_string(),
            provider_path: ProviderPath::classify(path),
            body,
            json,
            wants_stream,
            wants_bedrock_eventstream,
        }
    }
}

pub fn default_response(facts: &RequestFacts) -> SimulatedResponse {
    match facts.provider_path {
        ProviderPath::Health => SimulatedResponse::text(200, "text/plain; charset=utf-8", "ok"),
        ProviderPath::AnthropicMessages => {
            if facts.wants_stream {
                anthropic_sse()
            } else {
                anthropic_message()
            }
        }
        ProviderPath::OpenAiChatCompletions => {
            if facts.wants_stream {
                openai_chat_sse()
            } else {
                openai_chat()
            }
        }
        ProviderPath::OpenAiResponses => {
            if facts.wants_stream {
                openai_responses_sse()
            } else {
                openai_responses()
            }
        }
        ProviderPath::OpenAiConversations => conversation_collection(&facts.method),
        ProviderPath::OpenAiConversation => conversation_object(&facts.method, &facts.path),
        ProviderPath::OpenAiConversationItems => conversation_items(&facts.method, &facts.path),
        ProviderPath::OpenAiConversationItem => conversation_item(&facts.method, &facts.path),
        ProviderPath::BedrockInvoke | ProviderPath::BedrockConverse => bedrock_invoke(),
        ProviderPath::BedrockInvokeStream | ProviderPath::BedrockConverseStream => {
            if facts.wants_bedrock_eventstream {
                bedrock_eventstream()
            } else {
                anthropic_sse()
            }
        }
        ProviderPath::VertexRawPredict => vertex_predict(),
        ProviderPath::VertexStreamRawPredict => anthropic_sse(),
        ProviderPath::Generic => generic(&facts.path),
    }
}

fn anthropic_message() -> SimulatedResponse {
    SimulatedResponse::json(
        200,
        json!({
            "id": "msg_sim_0001",
            "type": "message",
            "role": "assistant",
            "model": "headroom-simulator-anthropic",
            "content": [{"type": "text", "text": "simulated anthropic response"}],
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {"input_tokens": 12, "output_tokens": 4}
        }),
    )
}

fn anthropic_sse() -> SimulatedResponse {
    let body = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_sim_stream\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"headroom-simulator-anthropic\",\"content\":[],\"stop_reason\":null,\"usage\":{\"input_tokens\":12,\"output_tokens\":0}}}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"simulated anthropic stream\"}}\n\n",
        "event: content_block_stop\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":4}}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n"
    );
    SimulatedResponse::text(200, "text/event-stream", body)
}

fn openai_chat() -> SimulatedResponse {
    SimulatedResponse::json(
        200,
        json!({
            "id": "chatcmpl-sim-0001",
            "object": "chat.completion",
            "created": 1,
            "model": "headroom-simulator-openai-chat",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "simulated openai chat response"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}
        }),
    )
}

fn openai_chat_sse() -> SimulatedResponse {
    let body = concat!(
        "data: {\"id\":\"chatcmpl-sim-stream\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"headroom-simulator-openai-chat\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-sim-stream\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"headroom-simulator-openai-chat\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"simulated openai chat stream\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-sim-stream\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"headroom-simulator-openai-chat\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]\n\n"
    );
    SimulatedResponse::text(200, "text/event-stream", body)
}

fn openai_responses() -> SimulatedResponse {
    SimulatedResponse::json(
        200,
        json!({
            "id": "resp_sim_0001",
            "object": "response",
            "created_at": 1,
            "model": "headroom-simulator-openai-responses",
            "status": "completed",
            "output": [{
                "id": "msg_sim_0001",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "simulated openai responses response"}]
            }],
            "usage": {"input_tokens": 12, "output_tokens": 5, "total_tokens": 17}
        }),
    )
}

fn openai_responses_sse() -> SimulatedResponse {
    let body = concat!(
        "event: response.created\n",
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_sim_stream\",\"status\":\"in_progress\",\"model\":\"headroom-simulator-openai-responses\"}}\n\n",
        "event: output_item.added\n",
        "data: {\"type\":\"output_item.added\",\"item\":{\"id\":\"msg_sim_stream\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[]}}\n\n",
        "event: output_text.delta\n",
        "data: {\"type\":\"output_text.delta\",\"item_id\":\"msg_sim_stream\",\"output_index\":0,\"content_index\":0,\"delta\":\"simulated openai responses stream\"}\n\n",
        "event: output_item.done\n",
        "data: {\"type\":\"output_item.done\",\"item\":{\"id\":\"msg_sim_stream\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"simulated openai responses stream\"}]}}\n\n",
        "event: response.completed\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_sim_stream\",\"status\":\"completed\",\"usage\":{\"input_tokens\":12,\"output_tokens\":5,\"total_tokens\":17}}}\n\n"
    );
    SimulatedResponse::text(200, "text/event-stream", body)
}

fn conversation_collection(method: &str) -> SimulatedResponse {
    match method {
        "POST" => SimulatedResponse::json(
            200,
            json!({"id":"conv_sim_0001","object":"conversation","metadata":{}}),
        ),
        _ => SimulatedResponse::json(
            200,
            json!({"object":"list","data":[{"id":"conv_sim_0001","object":"conversation"}]}),
        ),
    }
}

fn conversation_object(method: &str, path: &str) -> SimulatedResponse {
    let id = path.rsplit('/').next().unwrap_or("conv_sim_0001");
    if method == "DELETE" {
        SimulatedResponse::json(200, json!({"id": id, "deleted": true}))
    } else {
        SimulatedResponse::json(
            200,
            json!({"id": id, "object": "conversation", "metadata": {}}),
        )
    }
}

fn conversation_items(method: &str, path: &str) -> SimulatedResponse {
    let id = path
        .trim_end_matches("/items")
        .rsplit('/')
        .next()
        .unwrap_or("conv_sim_0001");
    if method == "POST" {
        SimulatedResponse::json(
            200,
            json!({"id":"item_sim_0001","object":"conversation.item","conversation_id": id}),
        )
    } else {
        SimulatedResponse::json(
            200,
            json!({"object":"list","data":[{"id":"item_sim_0001","object":"conversation.item","conversation_id": id}]}),
        )
    }
}

fn conversation_item(method: &str, path: &str) -> SimulatedResponse {
    let item_id = path.rsplit('/').next().unwrap_or("item_sim_0001");
    if method == "DELETE" {
        SimulatedResponse::json(200, json!({"id": item_id, "deleted": true}))
    } else {
        SimulatedResponse::json(200, json!({"id": item_id, "object": "conversation.item"}))
    }
}

fn bedrock_invoke() -> SimulatedResponse {
    SimulatedResponse::json(
        200,
        json!({
            "id": "msg_bedrock_sim_0001",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "simulated bedrock anthropic response"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 12, "output_tokens": 5}
        }),
    )
}

fn vertex_predict() -> SimulatedResponse {
    SimulatedResponse::json(
        200,
        json!({
            "id": "msg_vertex_sim_0001",
            "type": "message",
            "role": "assistant",
            "model": "headroom-simulator-vertex-anthropic",
            "content": [{"type": "text", "text": "simulated vertex anthropic response"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 12, "output_tokens": 5}
        }),
    )
}

fn generic(path: &str) -> SimulatedResponse {
    SimulatedResponse::json(
        200,
        json!({
            "id": "sim_generic_0001",
            "object": "headroom.simulator.response",
            "path": path,
            "message": "generic simulated response"
        }),
    )
}

fn bedrock_eventstream() -> SimulatedResponse {
    let payload = br#"{"type":"message_start","message":{"id":"msg_bedrock_stream","type":"message","role":"assistant","model":"headroom-simulator-bedrock","content":[],"stop_reason":null,"usage":{"input_tokens":12,"output_tokens":0}}}"#;
    let frame = encode_eventstream_message("chunk", payload);
    SimulatedResponse::text(200, "application/vnd.amazon.eventstream", frame)
}

fn encode_eventstream_message(event_type: &str, payload: &[u8]) -> Bytes {
    let mut headers = Vec::new();
    push_string_header(&mut headers, ":message-type", "event");
    push_string_header(&mut headers, ":event-type", event_type);
    push_string_header(&mut headers, ":content-type", "application/json");
    let total_len = 12 + headers.len() + payload.len() + 4;
    let headers_len = headers.len();
    let mut out = Vec::with_capacity(total_len);
    out.extend_from_slice(&(total_len as u32).to_be_bytes());
    out.extend_from_slice(&(headers_len as u32).to_be_bytes());
    let prelude_crc = crc32fast::hash(&out[..8]);
    out.extend_from_slice(&prelude_crc.to_be_bytes());
    out.extend_from_slice(&headers);
    out.extend_from_slice(payload);
    let message_crc = crc32fast::hash(&out);
    out.extend_from_slice(&message_crc.to_be_bytes());
    Bytes::from(out)
}

fn push_string_header(out: &mut Vec<u8>, name: &str, value: &str) {
    out.push(name.len() as u8);
    out.extend_from_slice(name.as_bytes());
    out.push(7);
    out.extend_from_slice(&(value.len() as u16).to_be_bytes());
    out.extend_from_slice(value.as_bytes());
}

fn header_contains(headers: &http::HeaderMap, name: &str, needle: &str) -> bool {
    headers
        .get(name)
        .and_then(|v| v.to_str().ok())
        .map(|v| {
            v.to_ascii_lowercase()
                .contains(&needle.to_ascii_lowercase())
        })
        .unwrap_or(false)
}

fn is_conversation(path: &str) -> bool {
    let parts: Vec<&str> = path.trim_matches('/').split('/').collect();
    parts.len() == 3 && parts[0] == "v1" && parts[1] == "conversations"
}

fn is_conversation_items(path: &str) -> bool {
    let parts: Vec<&str> = path.trim_matches('/').split('/').collect();
    parts.len() == 4 && parts[0] == "v1" && parts[1] == "conversations" && parts[3] == "items"
}

fn is_conversation_item(path: &str) -> bool {
    let parts: Vec<&str> = path.trim_matches('/').split('/').collect();
    parts.len() == 5 && parts[0] == "v1" && parts[1] == "conversations" && parts[3] == "items"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_supported_provider_paths() {
        assert_eq!(
            ProviderPath::classify("/v1/messages"),
            ProviderPath::AnthropicMessages
        );
        assert_eq!(
            ProviderPath::classify("/v1/chat/completions"),
            ProviderPath::OpenAiChatCompletions
        );
        assert_eq!(
            ProviderPath::classify("/v1/responses"),
            ProviderPath::OpenAiResponses
        );
        assert_eq!(
            ProviderPath::classify("/v1/conversations/c/items/i"),
            ProviderPath::OpenAiConversationItem
        );
        assert_eq!(
            ProviderPath::classify("/model/anthropic.claude-3-haiku/invoke-with-response-stream"),
            ProviderPath::BedrockInvokeStream
        );
        assert_eq!(
            ProviderPath::classify(
                "/v1beta1/projects/p/locations/us/publishers/anthropic/models/claude:rawPredict"
            ),
            ProviderPath::VertexRawPredict
        );
    }
}
