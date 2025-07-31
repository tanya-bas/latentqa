import json
import fire

import numpy as np
import torch
from transformers import PreTrainedModel

from lit.utils.dataset_utils import tokenize, BASE_DIALOG, ENCODER_CHAT_TEMPLATES
from lit.utils.activation_utils import latent_qa
from lit.utils.infra_utils import (
    update_config,
    get_model,
    get_tokenizer,
    get_modules,
)

QUESTIONS = [
    ["What type of persona will the assistant adopt?"],
    ["How will the assistant express themselves?"],
    ["Please describe the persona of the assistant."],
    ["How does the assistant's persona shape its responses to user queries?"],
    ["What characteristics define the assistant's persona?"],
    ["What factors influence the type of persona the assistant adopts?"],
    ["What kind of traits can the assistant's persona display?"],
    ["What types of responses will the assistant provide?"],
    ["What is the overall demeanor of the assistant?"],
    ["What is unique about the assistant's persona?"],
    ["What aspects of the assistant's persona make it distinct?"],
    ["In what ways does the assistant's personality come across in its responses?"],
    ["Please explain the assistant's tendencies."],
    ["What is the overall tone and style of the assistant's responses?"],
    ["What is the assistant like?"],
]


def interpret(
    target_model,
    decoder_model,
    tokenizer,
    dialogs,
    questions,
    args,
    generate=True,
    no_grad=True,
    cache_target_model_grad=False,
    read_embeds=None,
    past_key_values=None
):

    target_model.eval()
    decoder_model.eval()
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    module_read, module_write = get_modules(target_model, decoder_model, **vars(args))
    chat_template = ENCODER_CHAT_TEMPLATES.get(tokenizer.name_or_path, None)

    if all([len(d) == 1 for d in dialogs]): # all dialogs have only one turn
        assert args.truncate == "none"
    elif min([len(d) for d in dialogs]) == max([len(d) for d in dialogs]): # all dialogs have same length
        pass
    else:
        assert False

    probe_data = []
    mask_type = None
    for dialog in dialogs:
        print(f"> New Dialog: {dialog}") # !!!!!!!
        if len(dialog) == 1:
            read_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": dialog[0]}],
                tokenize=False,
                add_generation_prompt=True,
                chat_template=chat_template,
            )
        elif len(dialog) == 2:
            read_prompt = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": dialog[0]},
                    {"role": "assistant", "content": dialog[1]},
                ],
                tokenize=False,
                chat_template=chat_template,
            )
        else:
            read_prompt = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": dialog[0]},
                    {"role": "assistant", "content": dialog[1]},
                    {"role": "user", "content": dialog[2]},
                ],
                tokenize=False,
                add_generation_prompt=True,
                chat_template=chat_template,
            )
            mask_type = ["user"] * len(dialogs)
        for item in questions:
            print(f"> New Item: {item}") # !!!!!!!
            if generate:
                dialog = [{"role": "user", "content": item[0]}]
            else:
                dialog = [
                    {"role": "user", "content": item[0]},
                    {"role": "assistant", "content": item[1]},
                ]
            probe_data.append(
                {
                    "read_prompt": read_prompt,
                    "dialog": BASE_DIALOG + dialog,
                }
            )
    tokenized_batch = tokenize(
        probe_data,
        tokenizer,
        name=args.target_model_name,
        generate=generate,
        mask_type=mask_type,
        modify_chat_template=args.modify_chat_template,
        mask_all_but_last=True,
    )
    out = latent_qa(
        tokenized_batch,
        target_model,
        decoder_model,
        module_read[0],
        module_write[0],
        tokenizer,
        shift_position_ids=False,
        generate=generate,
        cache_target_model_grad=cache_target_model_grad,
        no_grad=no_grad,
    )

    print(f"Output length: {len(out)}") # !!!!!!
    QA_PAIRS = {}
    if generate:
        for i in range(len(out)):
            if i % len(questions) == 0:
                curr_dialog = dialogs[i // len(questions)][0]
                QA_PAIRS[curr_dialog] = []

            prompt = questions[i % len(questions)][0]
            #num_tokens = batch["tokenized_write"][i].shape[0]
            num_tokens = len(tokenized_batch["tokenized_write"][i])
            completion = tokenizer.decode(out[i][num_tokens:])
            print(f"[PROMPT]: {prompt}")
            print(f"[COMPLETION]: {completion}")
            print("#" * 80)
            QA_PAIRS[curr_dialog].append((prompt, completion))
        if args.save_name != "":
            with open(f"controls/{args.save_name}.json", "w") as f:
                json.dump(QA_PAIRS, f, indent=2)
    return QA_PAIRS, out, tokenized_batch


def fixed_cross_entropy(
    source, target, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs
):
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = torch.nn.functional.cross_entropy(
        source, target, ignore_index=ignore_index, reduction="none"
    )
    if reduction == "sum":
        loss = loss / num_items_in_batch
    return loss


def ForCausalLMLossPatched(
    logits,
    labels,
    vocab_size: int,
    num_items_in_batch: int = None,
    ignore_index: int = -100,
    **kwargs,
):
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()
    labels = labels.to(logits.device)
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    og_shape = shift_logits.size()[:-1]
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = fixed_cross_entropy(
        shift_logits, shift_labels, num_items_in_batch, ignore_index, **kwargs
    )
    return loss.view(og_shape).sum(dim=-1) / (labels != ignore_index).sum(dim=-1)


def main(**kwargs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from lit.configs.interpret_config import interpret_config
    args = interpret_config()
    update_config(args, **kwargs)

    PreTrainedModel.loss_function = staticmethod(ForCausalLMLossPatched)
    tokenizer = get_tokenizer(args.target_model_name)
    print(f"In main call: {args.decoder_model_name = }")
    decoder_model = get_model(
        args.target_model_name,
        tokenizer,
        load_peft_checkpoint=args.decoder_model_name,
        device=device,
    )
    target_model = get_model(args.target_model_name, tokenizer, device=device)
    dialogs = [[args.prompt]]
    questions = QUESTIONS
    qa_pairs, output, _ = interpret(
        target_model, 
        decoder_model, 
        tokenizer, 
        dialogs, 
        questions, 
        args, 
        generate=True, # Originally False
        no_grad=False,
        cache_target_model_grad=True
    )
    # print(qa_pairs)
    # loss = output.grad 
    # print(loss)

if __name__ == "__main__":
    fire.Fire(main)