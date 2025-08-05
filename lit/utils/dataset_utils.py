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
    "microsoft/DialoGPT-large": 1,
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
    "microsoft/DialoGPT-large": 1,
}

PAD_TOKEN_IDS = {
    "meta-llama/Meta-Llama-3-8B-Instruct": 128010,
    "meta-llama/Llama-3.1-8B-Instruct": 128010,
    "meta-llama/Meta-Llama-3-70B-Instruct": 128010,
    "mistralai/Ministral-8B-Instruct-2410": 999,
    "mistralai/Mistral-Small-24B-Instruct-2501": 2,
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": 128010,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": 151643,
    "microsoft/DialoGPT-large": 50256,
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
    "microsoft/DialoGPT-large": (
        50256,  # BOS token for GPT-2
        torch.tensor([50256]),  # Simple format for GPT-2 style
        torch.tensor([50256]),  # Simple format for GPT-2 style
        torch.tensor([50256]),  # Simple format for GPT-2 style
    ),
}

# Reasoning models need to encode their thought tokens
ENCODER_CHAT_TEMPLATES = {
    "meta-llama/Llama-3.1-8B-Instruct": "{% set loop_messages = messages %}{% for message in loop_messages %}{% set role = message['role'] %}{% set content = '<|start_header_id|>' + role + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}",
    "mistralai/Mistral-Small-24B-Instruct-2501": "{{- bos_token }}\n\n{%- for message in messages %}\n    {%- if message['role'] == 'user' %}\n        {{- '[INST]' + message['content'] + '[/INST]' }}\n    {%- elif message['role'] == 'system' %}\n        {{- '[SYSTEM_PROMPT]' + message['content'] + '[/SYSTEM_PROMPT]' }}\n    {%- elif message['role'] == 'assistant' %}\n        {{- message['content'] + eos_token }}\n    {%- else %}\n        {{- raise_exception('Only user, system and assistant roles are supported!') }}\n    {%- endif %}\n{%- endfor %}",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜tool▁calls▁begin｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '担任' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '担任' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + ''}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{{'<｜Assistant｜>' + content + ''}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜> 牡丹' + message['content'] + '牡丹'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n 牡丹' + message['content'] + '牡丹'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜><think>\\n'}}{% endif %}",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜tool▁calls▁begin｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '担任' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '担任' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + ''}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{{'<｜Assistant｜>' + content + ''}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜> 牡丹' + message['content'] + '牡丹'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n 牡丹' + message['content'] + '牡丹'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜><think>\\n'}}{% endif %}",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜Assistant｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '担任' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '担任' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + ''}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{{'<｜Assistant｜>' + content + ''}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜> 牡丹' + message['content'] + '牡丹'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n 牡丹' + message['content'] + '牡丹'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜><think>\\n'}}{% endif %}",
}
DECODER_CHAT_TEMPLATES = {
    "mistralai/Mistral-Small-24B-Instruct-2501": "{{- bos_token }}\n\n{%- for message in messages %}\n    {%- if message['role'] == 'user' %}\n        {{- '[INST]' + message['content'] + '[/INST]' }}\n    {%- elif message['role'] == 'system' %}\n        {{- '[SYSTEM_PROMPT]' + message['content'] + '[/SYSTEM_PROMPT]' }}\n    {%- elif message['role'] == 'assistant' %}\n        {{- message['content'] + eos_token }}\n    {%- else %}\n        {{- raise_exception('Only user, system and assistant roles are supported!') }}\n    {%- endif %}\n{%- endfor %}",
    "meta-llama/Meta-Llama-3-8B-Instruct": "{% set loop_messages = messages %}{% for message in loop_messages %}{% set role = message['role'] %}{% if role == 'assistant' %}{% set role = 'reflect' %}{% endif %}{% set content = '<|start_header_id|>' + role + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>reflect<|end_header_id|>\n\n' }}{% endif %}",
    "meta-llama/Llama-3.1-8B-Instruct": "{% set loop_messages = messages %}{% for message in loop_messages %}{% set role = message['role'] %}{% if role == 'assistant' %}{% set role = 'reflect' %}{% endif %}{% set content = '<|start_header_id|>' + role + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>reflect<|end_header_id|>\n\n' }}{% endif %}",
    "meta-llama/Meta-Llama-3-70B-Instruct": "{% set loop_messages = messages %}{% for message in loop_messages %}{% set role = message['role'] %}{% if role == 'assistant' %}{% set role = 'reflect' %}{% endif %}{% set content = '<|start_header_id|>' + role + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>reflect<|end_header_id|>\n\n' }}{% endif %}",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'REDACTED_SPECIAL_TOKEN' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<|reserved_special_token_8|>REDACTED_SPECIAL_TOKENREDACTED_SPECIAL_TOKEN' + tool['type'] + '担任' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + 'REDACTED_SPECIAL_TOKEN'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + 'REDACTED_SPECIAL_TOKEN' + tool['type'] + '担任' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + 'REDACTED_SPECIAL_TOKEN'}}{{'REDACTED_SPECIAL_TOKEN'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'REDACTED_SPECIAL_TOKEN' + message['content'] + ''}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}{{'<|reserved_special_token_8|>' + content + ''}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'REDACTED_SPECIAL_TOKEN 牡丹' + message['content'] + '牡丹'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n 牡丹' + message['content'] + '牡丹'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'REDACTED_SPECIAL_TOKEN'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<|reserved_special_token_8|><think>\\n'}}{% endif %}",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'REDACTED_SPECIAL_TOKEN' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<|reserved_special_token_8|>REDACTED_SPECIAL_TOKENREDACTED_SPECIAL_TOKEN' + tool['type'] + '担任' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + 'REDACTED_SPECIAL_TOKEN'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + 'REDACTED_SPECIAL_TOKEN' + tool['type'] + '担任' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + 'REDACTED_SPECIAL_TOKEN'}}{{'REDACTED_SPECIAL_TOKEN'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'REDACTED_SPECIAL_TOKEN' + message['content'] + ''}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}{{'<|reserved_special_token_8|>' + content + ''}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'REDACTED_SPECIAL_TOKEN 牡丹' + message['content'] + '牡丹'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n 牡丹' + message['content'] + '牡丹'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'REDACTED_SPECIAL_TOKEN'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<|reserved_special_token_8|><think>\\n'}}{% endif %}",
    "microsoft/DialoGPT-large": "{%- for message in messages %}{% if message['role'] == 'user' %}{{ message['content'] }}{% elif message['role'] == 'assistant' %}{{ message['content'] }}{% endif %}{% endfor %}",
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
    bos_token,start_tokens, end_tokens_default, end_tokens_modify = CHAT_FORMAT_TOKENS[
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
    return DistributedLengthBasedBatchSampler(
        dataset,
        train_config.batch_size_training,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
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
        batch_sampler=get_dist_batch_sampler(dataset_train, train_config, "train"),
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
            batch_sampler=get_dist_batch_sampler(dataset_eval, train_config, "val"),
        )
        return train_dataloader, eval_dataloader

    return train_dataloader, None
