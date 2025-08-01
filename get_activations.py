
import sys
sys.path.append('.') 
import json
import numpy as np
from lit.reading import main as reading_main, interpret, QUESTIONS
from lit.configs.interpret_config import interpret_config
from lit.utils.infra_utils import update_config, get_model, get_tokenizer
import torch
from transformers import PreTrainedModel


def load_models(decoder_model_name="aypan17/latentqa_llama-3-8b-instruct", 
                target_model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
    """Load models once and return them for reuse."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading models...")
    
    # Load tokenizer
    tokenizer = get_tokenizer(target_model_name)
    
    # Load decoder model
    decoder_model = get_model(
        target_model_name,
        tokenizer,
        load_peft_checkpoint=decoder_model_name,
        device=device,
    )
    
    # Load target model  
    target_model = get_model(target_model_name, tokenizer, device=device)
    
    print("Models loaded successfully!")
    return tokenizer, decoder_model, target_model


def get_activations_with_models(prompt_text, tokenizer, decoder_model, target_model):
    """Extract activations using pre-loaded models."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Setup config
    args = interpret_config()
    args.prompt = prompt_text
    
    # Setup dialogs
    dialogs = [[args.prompt]]
    
    # Run interpretation
    qa_pairs, activations, tokenized_batch, activation_cache = interpret(
        target_model, 
        decoder_model, 
        tokenizer, 
        dialogs, 
        QUESTIONS, 
        args, 
        generate=True,
        no_grad=False,
        cache_target_model_grad=True
    )
    
    return {
        "qa_pairs": qa_pairs,
        "activations": activations,
        "tokenized_batch": tokenized_batch,
        "activations_cache": activation_cache
    }


def load_data(file_path, exclude_categories=['o1', 'nq', 'sqa']):
    """Load LatentQA data from URL filtering out specified categories."""
    with open(file_path, 'r') as file:
        data = json.load(file)

    valid_data = []
    texts = []
    labels = []

    for item in data:
        if 'control_user' in item and 'label' in item:
            text = item['control_user']
            if text and text.strip():
                # Extract category to check for filtering
                label = item['label']
                parts = label.split('-')
                category = parts[1] if len(parts) > 1 else 'unknown'

                # Skip excluded categories
                if category not in exclude_categories:
                    valid_data.append(item)
                    texts.append(text.strip())
                    labels.append(label)

    # NOTE: valid_data is used only here which I found useful for debugging but we cna remove for cleanness later
    print(f"Loaded {len(data)} total entries, {len(valid_data)} valid entries after filtering")
    return texts, labels

if __name__ == "__main__":
    file_path = "/root/latentqa/data/eval/control.json"
    texts, labels = load_data(file_path)

    # Load models before processing
    tokenizer, decoder_model, target_model = load_models()
    all_activations = []
    all_labels_prompts = []
    
    for i in range(len(texts)):
        text = texts[i]
        label = labels[i]
        print(f"Processing {i+1}/{len(texts)}: {label}")
        
        result = get_activations_with_models(text, tokenizer, decoder_model, target_model)
        
        # activation cache (shape: [1, seq_len, 4096])
        activation_cache = result['activations_cache'][0]  #
        
        # NOTE: Here saving only last token activations becasue afaik they are most representative of final behavioral state but correct me if I am wrong
        last_token_activations = activation_cache[0, -1, :]  
        flattened_activations = last_token_activations.cpu().float().numpy()
        all_activations.append(flattened_activations)
        
        # Store label and prompt pair