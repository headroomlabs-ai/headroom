"""Independent literal provider events for socket and maintained-parser oracles."""

CHAT = """data: {"id":"chat_1","object":"chat.completion.chunk","created":1,"model":"fixture-model","choices":[{"index":0,"delta":{"role":"assistant","content":"雪","tool_calls":[{"index":0,"id":"tool_1","type":"function","function":{"name":"weather","arguments":"{\\"city\\":\\"雪\\"}"}}]},"finish_reason":null}]}

data: {"id":"chat_1","object":"chat.completion.chunk","created":1,"model":"fixture-model","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}

data: {"id":"chat_1","object":"chat.completion.chunk","created":1,"model":"fixture-model","choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}

data: [DONE]

""".replace("\n", "\r\n").encode()

RESPONSES = """event: response.output_text.delta
data: {"type":"response.output_text.delta","sequence_number":1,"item_id":"msg_1","output_index":0,"content_index":0,"delta":"雪","logprobs":[]}

event: response.function_call_arguments.delta
data: {"type":"response.function_call_arguments.delta","sequence_number":2,"item_id":"tool_1","output_index":1,"delta":"{\\"city\\":\\"雪\\"}"}

event: response.completed
data: {"type":"response.completed","sequence_number":3,"response":{"id":"resp_1","object":"response","created_at":1,"status":"completed","model":"fixture-model","output":[],"usage":{"input_tokens":3,"output_tokens":2,"total_tokens":5}}}

""".encode()

ANTHROPIC = """event: message_start
data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"fixture-model","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":3,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"雪"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"tool_1","name":"weather","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\\"city\\":\\"雪\\"}"}}

event: content_block_stop
data: {"type":"content_block_stop","index":1}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":2}}

event: message_stop
data: {"type":"message_stop"}

""".encode()

ANTHROPIC_TEXT = """event: message_start
data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"fixture-model","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":3,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"雪"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":2}}

event: message_stop
data: {"type":"message_stop"}

""".encode()

GEMINI = """data: {"candidates":[{"index":0,"content":{"role":"model","parts":[{"text":"雪"}]}},{"index":1,"content":{"role":"model","parts":[{"functionCall":{"name":"weather","args":{"city":"雪"}}}]}}]}

data: {"candidates":[{"index":0,"finishReason":"STOP"}]}

data: {"candidates":[{"index":1,"content":{"role":"model","parts":[{"text":"tail"}]}}]}

data: {"candidates":[{"index":1,"finishReason":"STOP"}]}

data: {"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":2,"totalTokenCount":5}}

""".encode()
