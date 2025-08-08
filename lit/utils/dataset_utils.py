import random
from itertools import islice
import json
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.distributed as dist

###################################
###### Tokens and formatting ######
###################################

IGNORE_IDX = -100

NUM_READ_TOKENS_TO_SHIFT = {
    "meta-llama/Meta-Llama-3-8B-Instruct": 1,
    "meta-llama/Llama-3.1-8B-Instruct": 1,
    "meta-llama/Meta-Llama-3-70B-Instruct": 1,
    "mistralai/Ministral-8B-Instruct-2410": 2,
    "mistralai/Mistral-Small-24B-Instruct-2501": 2,
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": 2,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": 2,
    "openai/gpt-oss-20b": 1,
}

# Magic numbers that are the length of the user tag + BOS token
NUM_WRITE_TOKENS_TO_SHIFT = {
    "meta-llama/Meta-Llama-3-8B-Instruct": 5,
    "meta-llama/Llama-3.1-8B-Instruct": 5,
    "meta-llama/Meta-Llama-3-70B-Instruct": 5,
    "mistralai/Ministral-8B-Instruct-2410": 2,
    "mistralai/Mistral-Small-24B-Instruct-2501": 2,
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": 2,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": 2,
    "openai/gpt-oss-20b": 3,
}

PAD_TOKEN_IDS = {
    "meta-llama/Meta-Llama-3-8B-Instruct": 128010,
    "meta-llama/Llama-3.1-8B-Instruct": 128010,
    "meta-llama/Meta-Llama-3-70B-Instruct": 128010,
    "mistralai/Ministral-8B-Instruct-2410": 999,
    "mistralai/Mistral-Small-24B-Instruct-2501": 2,
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": 128010,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": 151643,
    "openai/gpt-oss-20b": 199999,
}

# Magic numbers that correspond to the token idxs of the chat format for the models
# <bos>, <user>, <assistant>, <assistant_with_reflection>
CHAT_FORMAT_TOKENS = {
    "meta-llama/Meta-Llama-3-8B-Instruct": (
        128000,
        torch.tensor([128006, 882, 128007, 271]),
        torch.tensor([128006, 78191, 128007, 271]),
        torch.tensor([128006, 36013, 128007, 271]),
    ),
    "meta-llama/Llama-3.1-8B-Instruct": (
        128000,
        torch.tensor([128006, 882, 128007, 271]),
        torch.tensor([128006, 78191, 128007, 271]),
        torch.tensor([128006, 36013, 128007, 271]),
    ),
    "meta-llama/Meta-Llama-3-70B-Instruct": (
        128000,
        torch.tensor([128006, 882, 128007, 271]),
        torch.tensor([128006, 78191, 128007, 271]),
        torch.tensor([128006, 36013, 128007, 271]),
    ),
    "mistralai/Ministral-8B-Instruct-2410": (
        1,
        torch.tensor([3]),
        torch.tensor([4]),
        torch.tensor([4]),
    ),
    "mistralai/Mistral-Small-24B-Instruct-2501": (
        1,
        torch.tensor([3]),
        torch.tensor([4]),
        torch.tensor([4]),
    ),
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": (
        torch.tensor([128011]),
        torch.tensor([128012]),
        torch.tensor([128016]),
    ),
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": (
        torch.tensor([151644]),
        torch.tensor([151645]),
        torch.tensor([151665]),
    ),
    "openai/gpt-oss-20b": (
        27,  # BOS token (actual value from get_constants.py output)
        torch.tensor([200006, 1428, 200008]),  # <|start|>user<|message|> - User
        torch.tensor(
            [200006, 173781, 200008]
        ),  # <|start|>assistant<|message|> - Assistant
        torch.tensor(
            [200006, 76183, 200008]
        ),  # <|start|>reflect<|message|> - Assistant with reflection
    ),
}

# Reasoning models need to encode their thought tokens
ENCODER_CHAT_TEMPLATES = {
    "meta-llama/Llama-3.1-8B-Instruct": "{% set loop_messages = messages %}{% for message in loop_messages %}{% set role = message['role'] %}{% set content = '<|start_header_id|>' + role + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}",
    "mistralai/Mistral-Small-24B-Instruct-2501": "{{- bos_token }}\n\n{%- for message in messages %}\n    {%- if message['role'] == 'user' %}\n        {{- '[INST]' + message['content'] + '[/INST]' }}\n    {%- elif message['role'] == 'system' %}\n        {{- '[SYSTEM_PROMPT]' + message['content'] + '[/SYSTEM_PROMPT]' }}\n    {%- elif message['role'] == 'assistant' %}\n        {{- message['content'] + eos_token }}\n    {%- else %}\n        {{- raise_exception('Only user, system and assistant roles are supported!') }}\n    {%- endif %}\n{%- endfor %}",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜><think>\\n'}}{% endif %}",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜><think>\\n'}}{% endif %}",
    "openai/gpt-oss-20b": '{#-\n  In addition to the normal inputs of `messages` and `tools`, this template also accepts the\n  following kwargs:\n  - "builtin_tools": A list, can contain "browser" and/or "python".\n  - "model_identity": A string that optionally describes the model identity.\n  - "reasoning_effort": A string that describes the reasoning effort, defaults to "medium".\n #}\n\n{#- Tool Definition Rendering ============================= #}\n{%- macro render_typescript_type(param_spec, required_params, is_nullable=false) -%}\n    {%- if param_spec.type == "array" -%}\n        {%- if param_spec[\'items\'] -%}\n            {%- if param_spec[\'items\'][\'type\'] == "string" -%}\n                {{- "string[]" }}\n            {%- elif param_spec[\'items\'][\'type\'] == "number" -%}\n                {{- "number[]" }}\n            {%- elif param_spec[\'items\'][\'type\'] == "integer" -%}\n                {{- "number[]" }}\n            {%- elif param_spec[\'items\'][\'type\'] == "boolean" -%}\n                {{- "boolean[]" }}\n            {%- else -%}\n                {%- set inner_type = render_typescript_type(param_spec[\'items\'], required_params) -%}\n                {%- if inner_type == "object | object" or inner_type|length > 50 -%}\n                    {{- "any[]" }}\n                {%- else -%}\n                    {{- inner_type + "[]" }}\n                {%- endif -%}\n            {%- endif -%}\n            {%- if param_spec.nullable -%}\n                {{- " | null" }}\n            {%- endif -%}\n        {%- else -%}\n            {{- "any[]" }}\n            {%- if param_spec.nullable -%}\n                {{- " | null" }}\n            {%- endif -%}\n        {%- endif -%}\n    {%- elif param_spec.type is defined and param_spec.type is iterable and param_spec.type is not string and param_spec.type is not mapping and param_spec.type[0] is defined -%}\n        {#- Handle array of types like ["object", "object"] from Union[dict, list] #}\n        {%- if param_spec.type | length > 1 -%}\n            {{- param_spec.type | join(" | ") }}\n        {%- else -%}\n            {{- param_spec.type[0] }}\n        {%- endif -%}\n    {%- elif param_spec.oneOf -%}\n        {#- Handle oneOf schemas - check for complex unions and fallback to any #}\n        {%- set has_object_variants = false -%}\n        {%- for variant in param_spec.oneOf -%}\n            {%- if variant.type == "object" -%}\n                {%- set has_object_variants = true -%}\n            {%- endif -%}\n        {%- endfor -%}\n        {%- if has_object_variants and param_spec.oneOf|length > 1 -%}\n            {{- "any" }}\n        {%- else -%}\n            {%- for variant in param_spec.oneOf -%}\n                {{- render_typescript_type(variant, required_params) -}}\n                {%- if variant.description %}\n                    {{- "// " + variant.description }}\n                {%- endif -%}\n                {%- if variant.default is defined %}\n                    {{ "// default: " + variant.default|tojson }}\n                {%- endif -%}\n                {%- if not loop.last %}\n                    {{- " | " }}\n                {% endif -%}\n            {%- endfor -%}\n        {%- endif -%}\n    {%- elif param_spec.type == "string" -%}\n        {%- if param_spec.enum -%}\n            {{- \'"\' + param_spec.enum|join(\'" | "\') + \'"\' -}}\n        {%- else -%}\n            {{- "string" }}\n            {%- if param_spec.nullable %}\n                {{- " | null" }}\n            {%- endif -%}\n        {%- endif -%}\n    {%- elif param_spec.type == "number" -%}\n        {{- "number" }}\n    {%- elif param_spec.type == "integer" -%}\n        {{- "number" }}\n    {%- elif param_spec.type == "boolean" -%}\n        {{- "boolean" }}\n\n    {%- elif param_spec.type == "object" -%}\n        {%- if param_spec.properties -%}\n            {{- "{\\n" }}\n            {%- for prop_name, prop_spec in param_spec.properties.items() -%}\n                {{- prop_name -}}\n                {%- if prop_name not in (param_spec.required or []) -%}\n                    {{- "?" }}\n                {%- endif -%}\n                {{- ": " }}\n                {{ render_typescript_type(prop_spec, param_spec.required or []) }}\n                {%- if not loop.last -%}\n                    {{-", " }}\n                {%- endif -%}\n            {%- endfor -%}\n            {{- "}" }}\n        {%- else -%}\n            {{- "object" }}\n        {%- endif -%}\n    {%- else -%}\n        {{- "any" }}\n    {%- endif -%}\n{%- endmacro -%}\n\n{%- macro render_tool_namespace(namespace_name, tools) -%}\n    {{- "## " + namespace_name + "\\n\\n" }}\n    {{- "namespace " + namespace_name + " {\\n\\n" }}\n    {%- for tool in tools %}\n        {%- set tool = tool.function %}\n        {{- "// " + tool.description + "\\n" }}\n        {{- "type "+tool.name + " = " }}\n        {%- if tool.parameters and tool.parameters.properties %}\n            {{- "(_: {\\n" }}\n            {%- for param_name, param_spec in tool.parameters.properties.items() %}\n                {%- if param_spec.description %}\n                    {{- "// " + param_spec.description + "\\n" }}\n                {%- endif %}\n                {{- param_name }}\n                {%- if param_name not in (tool.parameters.required or []) -%}\n                    {{- "?" }}\n                {%- endif -%}\n                {{- ": " }}\n                {{- render_typescript_type(param_spec, tool.parameters.required or []) }}\n                {%- if param_spec.default is defined -%}\n                    {%- if param_spec.enum %}\n                        {{- ", // default: " + param_spec.default }}\n                    {%- elif param_spec.oneOf %}\n                        {{- "// default: " + param_spec.default }}\n                    {%- else %}\n                        {{- ", // default: " + param_spec.default|tojson }}\n                    {%- endif -%}\n                {%- endif -%}\n                {%- if not loop.last %}\n                    {{- ",\\n" }}\n                {%- else %}\n                    {{- ",\\n" }}\n                {%- endif -%}\n            {%- endfor %}\n            {{- "}) => any;\\n\\n" }}\n        {%- else -%}\n            {{- "() => any;\\n\\n" }}\n        {%- endif -%}\n    {%- endfor %}\n    {{- "} // namespace " + namespace_name }}\n{%- endmacro -%}\n\n{%- macro render_builtin_tools(browser_tool, python_tool) -%}\n    {%- if browser_tool %}\n        {{- "## browser\\n\\n" }}\n        {{- "// Tool for browsing.\\n" }}\n        {{- "// The `cursor` appears in brackets before each browsing display: `[{cursor}]`.\\n" }}\n        {{- "// Cite information from the tool using the following format:\\n" }}\n        {{- "// `【{cursor}†L{line_start}(-L{line_end})?】`, for example: `【6†L9-L11】` or `【8†L3】`.\\n" }}\n        {{- "// Do not quote more than 10 words directly from the tool output.\\n" }}\n        {{- "// sources=web (default: web)\\n" }}\n        {{- "namespace browser {\\n\\n" }}\n        {{- "// Searches for information related to `query` and displays `topn` results.\\n" }}\n        {{- "type search = (_: {\\n" }}\n        {{- "query: string,\\n" }}\n        {{- "topn?: number, // default: 10\\n" }}\n        {{- "source?: string,\\n" }}\n        {{- "}) => any;\\n\\n" }}\n        {{- "// Opens the link `id` from the page indicated by `cursor` starting at line number `loc`, showing `num_lines` lines.\\n" }}\n        {{- "// Valid link ids are displayed with the formatting: `【{id}†.*】`.\\n" }}\n        {{- "// If `cursor` is not provided, the most recent page is implied.\\n" }}\n        {{- "// If `id` is a string, it is treated as a fully qualified URL associated with `source`.\\n" }}\n        {{- "// If `loc` is not provided, the viewport will be positioned at the beginning of the document or centered on the most relevant passage, if available.\\n" }}\n        {{- "// Use this function without `id` to scroll to a new location of an opened page.\\n" }}\n        {{- "type open = (_: {\\n" }}\n        {{- "id?: number | string, // default: -1\\n" }}\n        {{- "cursor?: number, // default: -1\\n" }}\n        {{- "loc?: number, // default: -1\\n" }}\n        {{- "num_lines?: number, // default: -1\\n" }}\n        {{- "view_source?: boolean, // default: false\\n" }}\n        {{- "source?: string,\\n" }}\n        {{- "}) => any;\\n\\n" }}\n        {{- "// Finds exact matches of `pattern` in the current page, or the page given by `cursor`.\\n" }}\n        {{- "type find = (_: {\\n" }}\n        {{- "pattern: string,\\n" }}\n        {{- "cursor?: number, // default: -1\\n" }}\n        {{- "}) => any;\\n\\n" }}\n        {{- "} // namespace browser\\n\\n" }}\n    {%- endif -%}\n\n    {%- if python_tool %}\n        {{- "## python\\n\\n" }}\n        {{- "Use this tool to execute Python code in your chain of thought. The code will not be shown to the user. This tool should be used for internal reasoning, but not for code that is intended to be visible to the user (e.g. when creating plots, tables, or files).\\n\\n" }}\n        {{- "When you send a message containing Python code to python, it will be executed in a stateful Jupyter notebook environment. python will respond with the output of the execution or time out after 120.0 seconds. The drive at \'/mnt/data\' can be used to save and persist user files. Internet access for this session is UNKNOWN. Depends on the cluster.\\n\\n" }}\n    {%- endif -%}\n{%- endmacro -%}\n\n{#- System Message Construction ============================================ #}\n{%- macro build_system_message() -%}\n    {%- if model_identity is not defined %}\n        {%- set model_identity = "You are ChatGPT, a large language model trained by OpenAI." %}\n    {%- endif %}\n    {{- model_identity + "\\n" }}\n    {{- "Knowledge cutoff: 2024-06\\n" }}\n    {{- "Current date: " + strftime_now("%Y-%m-%d") + "\\n\\n" }}\n    {%- if reasoning_effort is not defined %}\n        {%- set reasoning_effort = "medium" %}\n    {%- endif %}\n    {{- "Reasoning: " + reasoning_effort + "\\n\\n" }}\n    {%- if builtin_tools %}\n        {{- "# Tools\\n\\n" }}\n        {%- set available_builtin_tools = namespace(browser=false, python=false) %}\n        {%- for tool in builtin_tools %}\n            {%- if tool == "browser" %}\n                {%- set available_builtin_tools.browser = true %}\n            {%- elif tool == "python" %}\n                {%- set available_builtin_tools.python = true %}\n            {%- endif %}\n        {%- endfor %}\n        {{- render_builtin_tools(available_builtin_tools.browser, available_builtin_tools.python) }}\n    {%- endif -%}\n    {{- "# Valid channels: analysis, commentary, final. Channel must be included for every message." }}\n    {%- if tools -%}\n        {{- "\\nCalls to these tools must go to the commentary channel: \'functions\'." }}\n    {%- endif -%}\n{%- endmacro -%}\n\n{#- Main Template Logic ================================================= #}\n{#- Set defaults #}\n\n{#- Render system message #}\n{{- "<|start|>system<|message|>" }}\n{{- build_system_message() }}\n{{- "<|end|>" }}\n\n{#- Extract developer message #}\n{%- if messages[0].role == "developer" or messages[0].role == "system" %}\n    {%- set developer_message = messages[0].content %}\n    {%- set loop_messages = messages[1:] %}\n{%- else %}\n    {%- set developer_message = "" %}\n    {%- set loop_messages = messages %}\n{%- endif %}\n\n{#- Render developer message #}\n{%- if developer_message or tools %}\n    {{- "<|start|>developer<|message|>" }}\n    {%- if developer_message %}\n        {{- "# Instructions\\n\\n" }}\n        {{- developer_message }}\n    {%- endif %}\n    {%- if tools -%}\n        {{- "\\n\\n" }}\n        {{- "# Tools\\n\\n" }}\n        {{- render_tool_namespace("functions", tools) }}\n    {%- endif -%}\n    {{- "<|end|>" }}\n{%- endif %}\n\n{#- Render messages #}\n{%- set last_tool_call = namespace(name=none) %}\n{%- for message in loop_messages -%}\n    {#- At this point only assistant/user/tool messages should remain #}\n    {%- if message.role == \'assistant\' -%}\n        {#- Checks to ensure the messages are being passed in the format we expect #}\n        {%- if "content" in message %}\n            {%- if "<|channel|>analysis<|message|>" in message.content or "<|channel|>final<|message|>" in message.content %}\n                {{- raise_exception("You have passed a message containing <|channel|> tags in the content field. Instead of doing this, you should pass analysis messages (the string between \'<|message|>\' and \'<|end|>\') in the \'thinking\' field, and final messages (the string between \'<|message|>\' and \'<|end|>\') in the \'content\' field.") }}\n            {%- endif %}\n        {%- endif %}\n        {%- if "thinking" in message %}\n            {%- if "<|channel|>analysis<|message|>" in message.thinking or "<|channel|>final<|message|>" in message.thinking %}\n                {{- raise_exception("You have passed a message containing <|channel|> tags in the thinking field. Instead of doing this, you should pass analysis messages (the string between \'<|message|>\' and \'<|end|>\') in the \'thinking\' field, and final messages (the string between \'<|message|>\' and \'<|end|>\') in the \'content\' field.") }}\n            {%- endif %}\n        {%- endif %}\n        {%- if "tool_calls" in message %}\n            {#- We need very careful handling here - we want to drop the tool call analysis message if the model #}\n            {#- has output a later <|final|> message, but otherwise we want to retain it. This is the only case #}\n            {#- when we render CoT/analysis messages in inference. #}\n            {%- set future_final_message = namespace(found=false) %}\n            {%- for future_message in loop_messages[loop.index:] %}\n                {%- if future_message.role == \'assistant\' and "tool_calls" not in future_message %}\n                    {%- set future_final_message.found = true %}\n                {%- endif %}\n            {%- endfor %}\n            {#- We assume max 1 tool call per message, and so we infer the tool call name #}\n            {#- in "tool" messages from the most recent assistant tool call name #}\n            {%- set tool_call = message.tool_calls[0] %}\n            {%- if tool_call.function %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {%- if message.content and message.thinking %}\n                {{- raise_exception("Cannot pass both content and thinking in an assistant message with tool calls! Put the analysis message in one or the other, but not both.") }}\n            {%- elif message.content and not future_final_message.found %}\n                {{- "<|start|>assistant<|channel|>analysis<|message|>" + message.content + "<|end|>" }}\n            {%- elif message.thinking and not future_final_message.found %}\n                {{- "<|start|>assistant<|channel|>analysis<|message|>" + message.thinking + "<|end|>" }}\n            {%- endif %}\n            {{- "<|start|>assistant to=" }}\n            {{- "functions." + tool_call.name + "<|channel|>commentary " }}\n            {{- (tool_call.content_type if tool_call.content_type is defined else "json") + "<|message|>" }}\n            {{- tool_call.arguments|tojson }}\n            {{- "<|call|>" }}\n            {%- set last_tool_call.name = tool_call.name %}\n        {%- elif loop.last and not add_generation_prompt %}\n            {#- Only render the CoT if the final turn is an assistant turn and add_generation_prompt is false #}\n            {#- This is a situation that should only occur in training, never in inference. #}\n            {%- if "thinking" in message %}\n                {{- "<|start|>assistant<|channel|>analysis<|message|>" + message.thinking + "<|end|>" }}\n            {%- endif %}\n            {#- <|return|> indicates the end of generation, but <|end|> does not #}\n            {#- <|return|> should never be an input to the model, but we include it as the final token #}\n            {#- when training, so the model learns to emit it. #}\n            {{- "<|start|>assistant<|channel|>final<|message|>" + message.content + "<|return|>" }}\n        {%- else %}\n            {#- CoT is dropped during all previous turns, so we never render it for inference #}\n            {{- "<|start|>assistant<|channel|>final<|message|>" + message.content + "<|end|>" }}\n            {%- set last_tool_call.name = none %}\n        {%- endif %}\n    {%- elif message.role == \'tool\' -%}\n        {%- if last_tool_call.name is none %}\n            {{- raise_exception("Message has tool role, but there was no previous assistant message with a tool call!") }}\n        {%- endif %}\n        {{- "<|start|>functions." + last_tool_call.name }}\n        {{- " to=assistant<|channel|>commentary<|message|>" + message.content|tojson + "<|end|>" }}\n    {%- elif message.role == \'user\' -%}\n        {{- "<|start|>user<|message|>" + message.content + "<|end|>" }}\n    {%- endif -%}\n{%- endfor -%}\n\n{#- Generation prompt #}\n{%- if add_generation_prompt -%}\n<|start|>assistant\n{%- endif -%}',
}
DECODER_CHAT_TEMPLATES = {
    "mistralai/Mistral-Small-24B-Instruct-2501": "{{- bos_token }}\n\n{%- for message in messages %}\n    {%- if message['role'] == 'user' %}\n        {{- '[INST]' + message['content'] + '[/INST]' }}\n    {%- elif message['role'] == 'system' %}\n        {{- '[SYSTEM_PROMPT]' + message['content'] + '[/SYSTEM_PROMPT]' }}\n    {%- elif message['role'] == 'assistant' %}\n        {{- message['content'] + eos_token }}\n    {%- else %}\n        {{- raise_exception('Only user, system and assistant roles are supported!') }}\n    {%- endif %}\n{%- endfor %}",
    "meta-llama/Meta-Llama-3-8B-Instruct": "{% set loop_messages = messages %}{% for message in loop_messages %}{% set role = message['role'] %}{% if role == 'assistant' %}{% set role = 'reflect' %}{% endif %}{% set content = '<|start_header_id|>' + role + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>reflect<|end_header_id|>\n\n' }}{% endif %}",
    "meta-llama/Llama-3.1-8B-Instruct": "{% set loop_messages = messages %}{% for message in loop_messages %}{% set role = message['role'] %}{% if role == 'assistant' %}{% set role = 'reflect' %}{% endif %}{% set content = '<|start_header_id|>' + role + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>reflect<|end_header_id|>\n\n' }}{% endif %}",
    "meta-llama/Meta-Llama-3-70B-Instruct": "{% set loop_messages = messages %}{% for message in loop_messages %}{% set role = message['role'] %}{% if role == 'assistant' %}{% set role = 'reflect' %}{% endif %}{% set content = '<|start_header_id|>' + role + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>reflect<|end_header_id|>\n\n' }}{% endif %}",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<|reserved_special_token_8|><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}{{'<|reserved_special_token_8|>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<|reserved_special_token_8|><think>\\n'}}{% endif %}",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<|reserved_special_token_8|><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}{{'<|reserved_special_token_8|>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<|reserved_special_token_8|><think>\\n'}}{% endif %}",
    "openai/gpt-oss-20b": '{#-\n  Decoder template for OpenAI GPT-OSS-20B model using Harmony format.\n  For decoder models, assistant responses should use \'reflect\' channel instead of \'final\'.\n #}\n\n{#- Same complex template structure as encoder, but modified for decoder usage #}\n{%- macro render_typescript_type(param_spec, required_params, is_nullable=false) -%}\n    {%- if param_spec.type == "array" -%}\n        {%- if param_spec[\'items\'] -%}\n            {%- if param_spec[\'items\'][\'type\'] == "string" -%}\n                {{- "string[]" }}\n            {%- elif param_spec[\'items\'][\'type\'] == "number" -%}\n                {{- "number[]" }}\n            {%- elif param_spec[\'items\'][\'type\'] == "integer" -%}\n                {{- "number[]" }}\n            {%- elif param_spec[\'items\'][\'type\'] == "boolean" -%}\n                {{- "boolean[]" }}\n            {%- else -%}\n                {%- set inner_type = render_typescript_type(param_spec[\'items\'], required_params) -%}\n                {%- if inner_type == "object | object" or inner_type|length > 50 -%}\n                    {{- "any[]" }}\n                {%- else -%}\n                    {{- inner_type + "[]" }}\n                {%- endif -%}\n            {%- endif -%}\n            {%- if param_spec.nullable -%}\n                {{- " | null" }}\n            {%- endif -%}\n        {%- else -%}\n            {{- "any[]" }}\n            {%- if param_spec.nullable -%}\n                {{- " | null" }}\n            {%- endif -%}\n        {%- endif -%}\n    {%- elif param_spec.type is defined and param_spec.type is iterable and param_spec.type is not string and param_spec.type is not mapping and param_spec.type[0] is defined -%}\n        {#- Handle array of types like ["object", "object"] from Union[dict, list] #}\n        {%- if param_spec.type | length > 1 -%}\n            {{- param_spec.type | join(" | ") }}\n        {%- else -%}\n            {{- param_spec.type[0] }}\n        {%- endif -%}\n    {%- elif param_spec.oneOf -%}\n        {#- Handle oneOf schemas - check for complex unions and fallback to any #}\n        {%- set has_object_variants = false -%}\n        {%- for variant in param_spec.oneOf -%}\n            {%- if variant.type == "object" -%}\n                {%- set has_object_variants = true -%}\n            {%- endif -%}\n        {%- endfor -%}\n        {%- if has_object_variants and param_spec.oneOf|length > 1 -%}\n            {{- "any" }}\n        {%- else -%}\n            {%- for variant in param_spec.oneOf -%}\n                {{- render_typescript_type(variant, required_params) -}}\n                {%- if variant.description %}\n                    {{- "// " + variant.description }}\n                {%- endif -%}\n                {%- if variant.default is defined %}\n                    {{ "// default: " + variant.default|tojson }}\n                {%- endif -%}\n                {%- if not loop.last %}\n                    {{- " | " }}\n                {% endif -%}\n            {%- endfor -%}\n        {%- endif -%}\n    {%- elif param_spec.type == "string" -%}\n        {%- if param_spec.enum -%}\n            {{- \'"\' + param_spec.enum|join(\'" | "\') + \'"\' -}}\n        {%- else -%}\n            {{- "string" }}\n            {%- if param_spec.nullable %}\n                {{- " | null" }}\n            {%- endif -%}\n        {%- endif -%}\n    {%- elif param_spec.type == "number" -%}\n        {{- "number" }}\n    {%- elif param_spec.type == "integer" -%}\n        {{- "number" }}\n    {%- elif param_spec.type == "boolean" -%}\n        {{- "boolean" }}\n    {%- elif param_spec.type == "object" -%}\n        {%- if param_spec.properties -%}\n            {{- "{\\n" }}\n            {%- for prop_name, prop_spec in param_spec.properties.items() -%}\n                {{- prop_name -}}\n                {%- if prop_name not in (param_spec.required or []) -%}\n                    {{- "?" }}\n                {%- endif -%}\n                {{- ": " }}\n                {{ render_typescript_type(prop_spec, param_spec.required or []) }}\n                {%- if not loop.last -%}\n                    {{-", " }}\n                {%- endif -%}\n            {%- endfor -%}\n            {{- "}" }}\n        {%- else -%}\n            {{- "object" }}\n        {%- endif -%}\n    {%- else -%}\n        {{- "any" }}\n    {%- endif -%}\n{%- endmacro -%}\n\n{%- macro render_tool_namespace(namespace_name, tools) -%}\n    {{- "## " + namespace_name + "\\n\\n" }}\n    {{- "namespace " + namespace_name + " {\\n\\n" }}\n    {%- for tool in tools %}\n        {%- set tool = tool.function %}\n        {{- "// " + tool.description + "\\n" }}\n        {{- "type "+tool.name + " = " }}\n        {%- if tool.parameters and tool.parameters.properties %}\n            {{- "(_: {\\n" }}\n            {%- for param_name, param_spec in tool.parameters.properties.items() %}\n                {%- if param_spec.description %}\n                    {{- "// " + param_spec.description + "\\n" }}\n                {%- endif %}\n                {{- param_name }}\n                {%- if param_name not in (tool.parameters.required or []) -%}\n                    {{- "?" }}\n                {%- endif -%}\n                {{- ": " }}\n                {{- render_typescript_type(param_spec, tool.parameters.required or []) }}\n                {%- if param_spec.default is defined -%}\n                    {%- if param_spec.enum %}\n                        {{- ", // default: " + param_spec.default }}\n                    {%- elif param_spec.oneOf %}\n                        {{- "// default: " + param_spec.default }}\n                    {%- else %}\n                        {{- ", // default: " + param_spec.default|tojson }}\n                    {%- endif -%}\n                {%- endif -%}\n                {%- if not loop.last %}\n                    {{- ",\\n" }}\n                {%- else %}\n                    {{- ",\\n" }}\n                {%- endif -%}\n            {%- endfor %}\n            {{- "}) => any;\\n\\n" }}\n        {%- else -%}\n            {{- "() => any;\\n\\n" }}\n        {%- endif -%}\n    {%- endfor %}\n    {{- "} // namespace " + namespace_name }}\n{%- endmacro -%}\n\n{%- macro render_builtin_tools(browser_tool, python_tool) -%}\n    {%- if browser_tool %}\n        {{- "## browser\\n\\n" }}\n        {{- "// Tool for browsing.\\n" }}\n        {{- "// The `cursor` appears in brackets before each browsing display: `[{cursor}]`.\\n" }}\n        {{- "// Cite information from the tool using the following format:\\n" }}\n        {{- "// `【{cursor}†L{line_start}(-L{line_end})?】`, for example: `【6†L9-L11】` or `【8†L3】`.\\n" }}\n        {{- "// Do not quote more than 10 words directly from the tool output.\\n" }}\n        {{- "// sources=web (default: web)\\n" }}\n        {{- "namespace browser {\\n\\n" }}\n        {{- "// Searches for information related to `query` and displays `topn` results.\\n" }}\n        {{- "type search = (_: {\\n" }}\n        {{- "query: string,\\n" }}\n        {{- "topn?: number, // default: 10\\n" }}\n        {{- "source?: string,\\n" }}\n        {{- "}) => any;\\n\\n" }}\n        {{- "// Opens the link `id` from the page indicated by `cursor` starting at line number `loc`, showing `num_lines` lines.\\n" }}\n        {{- "// Valid link ids are displayed with the formatting: `【{id}†.*】`.\\n" }}\n        {{- "// If `cursor` is not provided, the most recent page is implied.\\n" }}\n        {{- "// If `id` is a string, it is treated as a fully qualified URL associated with `source`.\\n" }}\n        {{- "// If `loc` is not provided, the viewport will be positioned at the beginning of the document or centered on the most relevant passage, if available.\\n" }}\n        {{- "// Use this function without `id` to scroll to a new location of an opened page.\\n" }}\n        {{- "type open = (_: {\\n" }}\n        {{- "id?: number | string, // default: -1\\n" }}\n        {{- "cursor?: number, // default: -1\\n" }}\n        {{- "loc?: number, // default: -1\\n" }}\n        {{- "num_lines?: number, // default: -1\\n" }}\n        {{- "view_source?: boolean, // default: false\\n" }}\n        {{- "source?: string,\\n" }}\n        {{- "}) => any;\\n\\n" }}\n        {{- "// Finds exact matches of `pattern` in the current page, or the page given by `cursor`.\\n" }}\n        {{- "type find = (_: {\\n" }}\n        {{- "pattern: string,\\n" }}\n        {{- "cursor?: number, // default: -1\\n" }}\n        {{- "}) => any;\\n\\n" }}\n        {{- "} // namespace browser\\n\\n" }}\n    {%- endif -%}\n\n    {%- if python_tool %}\n        {{- "## python\\n\\n" }}\n        {{- "Use this tool to execute Python code in your chain of thought. The code will not be shown to the user. This tool should be used for internal reasoning, but not for code that is intended to be visible to the user (e.g. when creating plots, tables, or files).\\n\\n" }}\n        {{- "When you send a message containing Python code to python, it will be executed in a stateful Jupyter notebook environment. python will respond with the output of the execution or time out after 120.0 seconds. The drive at \'/mnt/data\' can be used to save and persist user files. Internet access for this session is UNKNOWN. Depends on the cluster.\\n\\n" }}\n    {%- endif -%}\n{%- endmacro -%}\n\n{#- System Message Construction ============================================ #}\n{%- macro build_system_message() -%}\n    {%- if model_identity is not defined %}\n        {%- set model_identity = "You are ChatGPT, a large language model trained by OpenAI." %}\n    {%- endif %}\n    {{- model_identity + "\\n" }}\n    {{- "Knowledge cutoff: 2024-06\\n" }}\n    {{- "Current date: " + strftime_now("%Y-%m-%d") + "\\n\\n" }}\n    {%- if reasoning_effort is not defined %}\n        {%- set reasoning_effort = "medium" %}\n    {%- endif %}\n    {{- "Reasoning: " + reasoning_effort + "\\n\\n" }}\n    {%- if builtin_tools %}\n        {{- "# Tools\\n\\n" }}\n        {%- set available_builtin_tools = namespace(browser=false, python=false) %}\n        {%- for tool in builtin_tools %}\n            {%- if tool == "browser" %}\n                {%- set available_builtin_tools.browser = true %}\n            {%- elif tool == "python" %}\n                {%- set available_builtin_tools.python = true %}\n            {%- endif %}\n        {%- endfor %}\n        {{- render_builtin_tools(available_builtin_tools.browser, available_builtin_tools.python) }}\n    {%- endif -%}\n    {{- "# Valid channels: analysis, commentary, final. Channel must be included for every message." }}\n    {%- if tools -%}\n        {{- "\\nCalls to these tools must go to the commentary channel: \'functions\'." }}\n    {%- endif -%}\n{%- endmacro -%}\n\n{#- Main Template Logic ================================================= #}\n{#- Set defaults #}\n\n{#- Render system message #}\n{{- "<|start|>system<|message|>" }}\n{{- build_system_message() }}\n{{- "<|end|>" }}\n\n{#- Extract developer message #}\n{%- if messages[0].role == "developer" or messages[0].role == "system" %}\n    {%- set developer_message = messages[0].content %}\n    {%- set loop_messages = messages[1:] %}\n{%- else %}\n    {%- set developer_message = "" %}\n    {%- set loop_messages = messages %}\n{%- endif %}\n\n{#- Render developer message #}\n{%- if developer_message or tools %}\n    {{- "<|start|>developer<|message|>" }}\n    {%- if developer_message %}\n        {{- "# Instructions\\n\\n" }}\n        {{- developer_message }}\n    {%- endif %}\n    {%- if tools -%}\n        {{- "\\n\\n" }}\n        {{- "# Tools\\n\\n" }}\n        {{- render_tool_namespace("functions", tools) }}\n    {%- endif -%}\n    {{- "<|end|>" }}\n{%- endif %}\n\n{#- Render messages #}\n{%- set last_tool_call = namespace(name=none) %}\n{%- for message in loop_messages -%}\n    {#- At this point only assistant/user/tool messages should remain #}\n    {%- if message.role == \'assistant\' -%}\n        {#- For decoder, use \'reflect\' channel instead of \'final\' #}\n        {%- set channel = \'reflect\' -%}\n        {#- Checks to ensure the messages are being passed in the format we expect #}\n        {%- if "content" in message %}\n            {%- if "<|channel|>analysis<|message|>" in message.content or "<|channel|>final<|message|>" in message.content %}\n                {{- raise_exception("You have passed a message containing <|channel|> tags in the content field. Instead of doing this, you should pass analysis messages (the string between \'<|message|>\' and \'<|end|>\') in the \'thinking\' field, and final messages (the string between \'<|message|>\' and \'<|end|>\') in the \'content\' field.") }}\n            {%- endif %}\n        {%- endif %}\n        {%- if "thinking" in message %}\n            {%- if "<|channel|>analysis<|message|>" in message.thinking or "<|channel|>final<|message|>" in message.thinking %}\n                {{- raise_exception("You have passed a message containing <|channel|> tags in the thinking field. Instead of doing this, you should pass analysis messages (the string between \'<|message|>\' and \'<|end|>\') in the \'thinking\' field, and final messages (the string between \'<|message|>\' and \'<|end|>\') in the \'content\' field.") }}\n            {%- endif %}\n        {%- endif %}\n        {%- if "tool_calls" in message %}\n            {#- We need very careful handling here - we want to drop the tool call analysis message if the model #}\n            {#- has output a later <|final|> message, but otherwise we want to retain it. This is the only case #}\n            {#- when we render CoT/analysis messages in inference. #}\n            {%- set future_final_message = namespace(found=false) %}\n            {%- for future_message in loop_messages[loop.index:] %}\n                {%- if future_message.role == \'assistant\' and "tool_calls" not in future_message %}\n                    {%- set future_final_message.found = true %}\n                {%- endif %}\n            {%- endfor %}\n            {#- We assume max 1 tool call per message, and so we infer the tool call name #}\n            {#- in "tool" messages from the most recent assistant tool call name #}\n            {%- set tool_call = message.tool_calls[0] %}\n            {%- if tool_call.function %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {%- if message.content and message.thinking %}\n                {{- raise_exception("Cannot pass both content and thinking in an assistant message with tool calls! Put the analysis message in one or the other, but not both.") }}\n            {%- elif message.content and not future_final_message.found %}\n                {{- "<|start|>assistant<|channel|>analysis<|message|>" + message.content + "<|end|>" }}\n            {%- elif message.thinking and not future_final_message.found %}\n                {{- "<|start|>assistant<|channel|>analysis<|message|>" + message.thinking + "<|end|>" }}\n            {%- endif %}\n            {{- "<|start|>assistant to=" }}\n            {{- "functions." + tool_call.name + "<|channel|>commentary " }}\n            {{- (tool_call.content_type if tool_call.content_type is defined else "json") + "<|message|>" }}\n            {{- tool_call.arguments|tojson }}\n            {{- "<|call|>" }}\n            {%- set last_tool_call.name = tool_call.name %}\n        {%- elif loop.last and not add_generation_prompt %}\n            {#- Only render the CoT if the final turn is an assistant turn and add_generation_prompt is false #}\n            {#- This is a situation that should only occur in training, never in inference. #}\n            {%- if "thinking" in message %}\n                {{- "<|start|>assistant<|channel|>analysis<|message|>" + message.thinking + "<|end|>" }}\n            {%- endif %}\n            {#- <|return|> indicates the end of generation, but <|end|> does not #}\n            {#- <|return|> should never be an input to the model, but we include it as the final token #}\n            {#- when training, so the model learns to emit it. #}\n            {{- "<|start|>assistant<|channel|>" + channel + "<|message|>" + message.content + "<|return|>" }}\n        {%- else %}\n            {#- CoT is dropped during all previous turns, so we never render it for inference #}\n            {{- "<|start|>assistant<|channel|>" + channel + "<|message|>" + message.content + "<|end|>" }}\n            {%- set last_tool_call.name = none %}\n        {%- endif %}\n    {%- elif message.role == \'tool\' -%}\n        {%- if last_tool_call.name is none %}\n            {{- raise_exception("Message has tool role, but there was no previous assistant message with a tool call!") }}\n        {%- endif %}\n        {{- "<|start|>functions." + last_tool_call.name }}\n        {{- " to=assistant<|channel|>commentary<|message|>" + message.content|tojson + "<|end|>" }}\n    {%- elif message.role == \'user\' -%}\n        {{- "<|start|>user<|message|>" + message.content + "<|end|>" }}\n    {%- endif -%}\n{%- endfor -%}\n\n{#- Generation prompt #}\n{%- if add_generation_prompt -%}\n<|start|>assistant\n{%- endif -%}',
}

# Dialog formats for the dataset
BASE_DIALOG = [
    {
        "role": "assistant",
        "content": "Sure, I've analyzed the assistant.",
    }
]

################################
###### Activation masking ######
################################


def mask_inputs(
    input_ids,
    tokenizer_name,
    mask_type=None,
    shift_start=False,
    mask_all_but_last=False,
    modify_chat_template=False,
):
    bos_token, start_tokens, end_tokens_default, end_tokens_modify = CHAT_FORMAT_TOKENS[
        tokenizer_name
    ]
    end_tokens = end_tokens_modify if modify_chat_template else end_tokens_default
    batch_size, seq_len = input_ids.shape
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for b in range(batch_size):
        start_idx = []
        end_idx = []
        for i in range(seq_len):
            if torch.equal(input_ids[b][i : i + len(start_tokens)], start_tokens):
                start_idx.append(i)
            if torch.equal(input_ids[b][i : i + len(end_tokens)], end_tokens):
                end_idx.append(i)

        if mask_type is None:
            if len(start_idx) != len(end_idx):
                # Data is improperly formatted so mask everything and skip this item
                mask[b][:] = True
                continue

            if mask_all_but_last:
                mask[b][: end_idx[-1] + len(end_tokens)] = True
            else:
                for i, (start, end) in enumerate(zip(start_idx, end_idx)):
                    if shift_start and i == 0:
                        mask[b][start - 1 : end + len(end_tokens)] = True
                    else:
                        mask[b][start : end + len(end_tokens)] = True
        elif mask_type[b] == "user":
            if len(start_idx) == 1:
                continue
            mask[b][start_idx[0] : start_idx[1]] = True
        elif mask_type[b] == "system":
            bos_idx = input_ids[b].tolist().index(bos_token)
            mask[b][bos_idx + 1 : start_idx[0]] = True
        else:
            raise ValueError(f"Invalid verb mask: {mask_type[b]}")
    return mask


def tokenize(
    batch,
    tokenizer,
    name=None,
    generate=False,
    mask_type=None,
    mask_all_but_last=False,
    modify_chat_template=False,
):
    name = tokenizer.name_or_path if name is None else name

    # Tokenize read inputs
    tokenized_read = tokenizer(
        [item["read_prompt"] for item in batch],
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    tokenized_batch = {"tokenized_read": tokenized_read}

    # Compute length of read input and maybe add verb_lengths
    read_lengths = torch.sum(tokenized_read.attention_mask, dim=1)
    tokenized_batch["read_lengths"] = read_lengths - 1  # Exclude BOS token

    if mask_type is not None:
        verb_mask = mask_inputs(tokenized_read.input_ids, name, mask_type=mask_type)
        verb_lengths = torch.sum(verb_mask, dim=1)
        pad_lengths = read_lengths - verb_lengths
        tokenized_batch["verb_lengths"] = verb_lengths
    else:
        pad_lengths = read_lengths

    # Tokenize dialog inputs
    queries = []
    for i in range(len(pad_lengths)):
        query = [
            {
                "role": "user",
                "content": "? " * (pad_lengths[i] - NUM_READ_TOKENS_TO_SHIFT[name]),
            }
        ]
        query += batch[i]["dialog"]
        queries.append(
            tokenizer.apply_chat_template(
                query,
                tokenize=False,
                add_generation_prompt=generate,
                chat_template=(
                    DECODER_CHAT_TEMPLATES[name] if modify_chat_template else None
                ),
            )
        )
    tokenized_write = tokenizer(
        queries,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    tokenized_batch["tokenized_write"] = tokenized_write

    # Compute length of write input
    write_lengths = torch.sum(tokenized_write.attention_mask, dim=1)
    tokenized_batch["write_lengths"] = write_lengths - NUM_WRITE_TOKENS_TO_SHIFT[name]

    # Add labels for training
    if not generate:
        user_inputs_mask = mask_inputs(
            tokenized_write.input_ids,
            name,
            mask_type=None,
            shift_start=any(
                [
                    m in name.lower()
                    for m in ["mistral", "llama-3", "deepseek-r1-distill"]
                ]
            ),
            mask_all_but_last=mask_all_but_last,
            modify_chat_template=modify_chat_template,
        )
        assert tokenizer.padding_side == "left"
        tokenized_write["labels"] = tokenized_write.input_ids.clone()
        mask = (tokenized_write.attention_mask == 0) | user_inputs_mask
        tokenized_write["labels"][mask] = IGNORE_IDX
    return tokenized_batch


###########################
####### Dataloading #######
###########################


class LatentQADataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_system,
        data_stimulus_completion,
        data_stimulus,
        data_control,
        qa_data,
        add_thought_tokens=False,
    ):
        self.tokenizer = tokenizer
        self.data = [
            data_system[0],
            data_stimulus_completion[0],
            data_stimulus[0],
            data_control[0],
        ]
        self.id_tuples = [
            data_system[1],
            data_stimulus_completion[1],
            data_stimulus[1],
            data_control[1],
        ]
        self.labels = [
            list(data_system[0].keys()),
            list(data_stimulus_completion[0].keys()),
            list(data_stimulus[0].keys()),
            list(data_control[0].keys()),
        ]
        self.qa_data = qa_data
        self.add_thought_tokens = add_thought_tokens
        self.chat_template = ENCODER_CHAT_TEMPLATES.get(tokenizer.name_or_path, None)
        self.lengths = []
        for idx in range(self.__len__()):
            behavior, qa = self.get_behavior_qa(idx)
            self.lengths.append(
                sum([len(s) for s in behavior]) + len(qa[0]) + len(qa[1])
            )

    def get_behavior_qa(self, idx):
        if idx < len(self.id_tuples[0]):
            j = 0
        elif idx < len(self.id_tuples[0]) + len(self.id_tuples[1]):
            j = 1
            idx -= len(self.id_tuples[0])
        elif idx < len(self.id_tuples[0]) + len(self.id_tuples[1]) + len(
            self.id_tuples[2]
        ):
            j = 2
            idx -= len(self.id_tuples[0]) + len(self.id_tuples[1])
        else:
            j = 3
            idx -= (
                len(self.id_tuples[0]) + len(self.id_tuples[1]) + len(self.id_tuples[2])
            )
        label_idx, data_idx, qa_idx = self.id_tuples[j][idx]
        label = self.labels[j][label_idx]
        return self.data[j][label][data_idx], self.qa_data[label][qa_idx]

    def __len__(self):
        return sum([len(id_tuples) for id_tuples in self.id_tuples])

    def __getitem__(self, idx):
        behavior, qa = self.get_behavior_qa(idx)
        (
            system,
            control_user,
            control_thought,
            control_model,
            stimulus_user,
            stimulus_thought,
            stimulus_model,
        ) = behavior
        if system != "":
            assert control_user == control_model == stimulus_model == ""
            read_prompt = [
                {"role": "system", "content": system},
                {"role": "user", "content": stimulus_user},
            ]
            add_generation_prompt = True
            mask_type = "system"
        elif control_model == "":
            assert stimulus_user == stimulus_model == ""
            read_prompt = [{"role": "user", "content": control_user}]
            add_generation_prompt = True
            mask_type = "user"
        elif stimulus_model == "":
            read_prompt = [
                {"role": "user", "content": control_user},
                {"role": "assistant", "content": control_model},
                {"role": "user", "content": stimulus_user},
            ]
            add_generation_prompt = True
            mask_type = "user"
        else:
            if self.add_thought_tokens:
                read_prompt = [
                    {"role": "user", "content": control_user},
                    {"role": "assistant", "content": control_model},
                    {"role": "user", "content": stimulus_user},
                    {
                        "role": "assistant",
                        "content": f"<think>\n{stimulus_thought}\n</think>\n\n{stimulus_model}",
                    },
                ]
                add_generation_prompt = False
                mask_type = "user"
            else:
                read_prompt = [
                    {"role": "user", "content": control_user},
                    {"role": "assistant", "content": control_model},
                    {"role": "user", "content": stimulus_user},
                    {"role": "assistant", "content": stimulus_model},
                ]
                add_generation_prompt = False
                mask_type = "user"
        read_prompt = self.tokenizer.apply_chat_template(
            read_prompt,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            chat_template=self.chat_template,
        )
        qa_dialog = [
            {"role": "user", "content": qa[0]},
            {"role": "assistant", "content": qa[1]},
        ]
        return {
            "read_prompt": read_prompt,
            "dialog": BASE_DIALOG + qa_dialog,
            "mask_type": mask_type,
        }


class DataCollatorForLatentQA:
    def __init__(
        self,
        tokenizer,
        generate=False,
        mask_all_but_last=False,
        nudge_persona=False,
        modify_chat_template=False,
    ):
        self.tokenizer = tokenizer
        self.generate = generate
        self.mask_all_but_last = mask_all_but_last
        self.nudge = "Base your answers on my instructions. " if nudge_persona else ""
        self.modify_chat_template = modify_chat_template

    def __call__(self, batch):
        formatted_batch = []
        mask_type = []
        for item in batch:
            formatted_batch.append(
                {
                    "read_prompt": item["read_prompt"],
                    "dialog": item["dialog"],
                    "label": item["dialog"][-1]["content"],
                }
            )
            mask_type.append(item["mask_type"])
        return tokenize(
            formatted_batch,
            self.tokenizer,
            mask_type=mask_type,
            generate=self.generate,
            mask_all_but_last=self.mask_all_but_last,
            modify_chat_template=self.modify_chat_template,
        )


class LengthBasedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(
        self, data_source, batch_size: int, drop_last: bool, shuffle: bool = True
    ) -> None:
        self.lengths = data_source.lengths
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

    def __iter__(self):
        ids = np.argsort(self.lengths)
        if self.drop_last:
            ids = ids[: len(ids) // self.batch_size * self.batch_size]

        batches = [
            ids[i : i + self.batch_size] for i in range(0, len(ids), self.batch_size)
        ]

        if self.shuffle:
            random.shuffle(batches)

        for b in batches:
            yield b

    def __len__(self):
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        else:
            return len(self.lengths) // self.batch_size + (
                len(self.lengths) % self.batch_size > 0
            )


class DistributedLengthBasedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(
        self,
        data_source,
        batch_size: int,
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        random.seed(seed)
        self.batch_sampler = LengthBasedBatchSampler(
            data_source, batch_size=batch_size, drop_last=True, shuffle=shuffle
        )
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        max_length = len(self.batch_sampler) // self.num_replicas * self.num_replicas
        return islice(self.batch_sampler, self.rank, max_length, self.num_replicas)

    def __len__(self):
        return len(self.batch_sampler) // self.num_replicas


def get_batch_sampler(dataset, train_config, mode):
    return LengthBasedBatchSampler(
        dataset,
        train_config.batch_size_training,
        drop_last=False,
        shuffle=(mode == "train"),
    )


def get_dist_batch_sampler(dataset, train_config, mode):
    # Check if distributed training is initialized
    try:
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    except (RuntimeError, ValueError):
        # Not in distributed mode, use single GPU settings
        world_size = 1
        rank = 0

    return DistributedLengthBasedBatchSampler(
        dataset,
        train_config.batch_size_training,
        num_replicas=world_size,
        rank=rank,
        shuffle=(mode == "train"),
        seed=train_config.seed,
    )


def get_dataset(train_config, tokenizer, train=True):
    FILTER = train_config.filter.split("-")
    qa_path = train_config.train_qa if train else train_config.eval_qa
    with open(qa_path, "r") as f:
        qa_data = json.load(f)
        NUM_QA = max([len(qa_data[label]) for label in qa_data])
        assert NUM_QA == min([len(qa_data[label]) for label in qa_data])

    def build_data_and_idx(path):
        # Get data
        data = defaultdict(list)
        if path == "":
            return data, []
        with open(path, "r") as f:
            raw_data = json.load(f)
            for item in raw_data:
                if item["label"].split("-")[0] in FILTER:
                    continue
                data[item["label"]].append(
                    (
                        item.get("system", ""),
                        item.get("control_user", ""),
                        item.get("control_thought", ""),
                        item.get("control_model", ""),
                        item.get("stimulus_user", ""),
                        item.get("stimulus_thought", ""),
                        item.get("stimulus_model", ""),
                    )
                )
        # Get id tuples
        NUM_BEHAVIORS = max([len(data[label]) for label in data])
        assert NUM_BEHAVIORS == min([len(data[label]) for label in data])
        id_tuples = range(len(data) * NUM_BEHAVIORS * NUM_QA)
        if train_config.train_percent == 1 or not train:
            id_tuples = list(id_tuples)
        else:
            id_tuples = random.sample(
                id_tuples, int(len(id_tuples) * train_config.train_percent)
            )
        for i in range(len(id_tuples)):
            label_idx = id_tuples[i] // (NUM_BEHAVIORS * NUM_QA)
            data_idx = (id_tuples[i] // NUM_QA) % NUM_BEHAVIORS
            qa_idx = id_tuples[i] % NUM_QA
            id_tuples[i] = (label_idx, data_idx, qa_idx)
        return data, id_tuples

    p0 = train_config.train_system if train else train_config.eval_system
    p1 = (
        train_config.train_stimulus_completion
        if train
        else train_config.eval_stimulus_completion
    )
    p2 = train_config.train_stimulus if train else train_config.eval_stimulus
    p3 = train_config.train_control if train else train_config.eval_control
    data_system = build_data_and_idx(p0)
    data_stimulus_completion = build_data_and_idx(p1)
    data_stimulus = build_data_and_idx(p2)
    data_control = build_data_and_idx(p3)

    return LatentQADataset(
        tokenizer,
        data_system,
        data_stimulus_completion,
        data_stimulus,
        data_control,
        qa_data,
        add_thought_tokens=train_config.add_thought_tokens,
    )


def get_dataloaders(train_config, tokenizer):
    # Check if we're in distributed mode
    try:
        dist.get_world_size()
        use_distributed = True
    except (RuntimeError, ValueError):
        use_distributed = False

    # Choose appropriate batch sampler based on distributed mode
    sampler_fn = get_dist_batch_sampler if use_distributed else get_batch_sampler

    dataset_train = get_dataset(train_config, tokenizer, train=True)
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        collate_fn=DataCollatorForLatentQA(
            tokenizer,
            mask_all_but_last=False,
            nudge_persona=train_config.nudge_persona,
            modify_chat_template=train_config.modify_chat_template,
        ),
        batch_sampler=sampler_fn(dataset_train, train_config, "train"),
    )
    if train_config.eval_ppl:
        dataset_eval = get_dataset(train_config, tokenizer, train=False)
        eval_dataloader = torch.utils.data.DataLoader(
            dataset_eval,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            collate_fn=DataCollatorForLatentQA(
                tokenizer,
                mask_all_but_last=False,
                nudge_persona=train_config.nudge_persona,
                modify_chat_template=train_config.modify_chat_template,
            ),
            batch_sampler=sampler_fn(dataset_eval, train_config, "val"),
        )
        return train_dataloader, eval_dataloader

    return train_dataloader, None
