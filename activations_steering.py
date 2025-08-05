import torch
import random
import numpy as np
from lit.utils.infra_utils import get_model, get_tokenizer

def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    # Make operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def extract_steering_vector(model, tokenizer, prompt, layer=15):
    """Extract activations from specified layer"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[layer]

def run_all_tests():
    set_seed(11)
    target_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    test_prompt = "How should I spend my ten dollars?"
    
    tokenizer = get_tokenizer(target_model_name)
    model = get_model(target_model_name, tokenizer, device="cuda")
    model = model.half()  # Convert to float16 to fix BFloat16 compatibility issue
    
    # ActAdd methodology based steering
    vegan_prompt = "Plant-based"
    meat_prompt = "Meat-based"
    
    vegan_vector = extract_steering_vector(model, tokenizer, vegan_prompt)
    meat_vector = extract_steering_vector(model, tokenizer, meat_prompt)
    
    min_len = min(vegan_vector.shape[1], meat_vector.shape[1])
    contrast_vector = vegan_vector[:, :min_len, :] - meat_vector[:, :min_len, :]
    
    print(f"Prompt: {test_prompt}")
    print("-" * 50)
    
    baseline = generate_with_model(model, tokenizer, test_prompt)
    simple = generate_with_steering(model, tokenizer, test_prompt, vegan_vector)
    contrast = generate_with_steering(model, tokenizer, test_prompt, contrast_vector)
    
    print(f"Baseline:  {baseline}")
    print("-" * 50)
    print(f"Simple:    {simple}")
    print("-" * 50)
    print(f"Contrast:  {contrast}")


def generate_with_model(model, tokenizer, prompt, max_new_tokens=200):
    """Simple generation without steering"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, 
                                do_sample=True, temperature=0.3, top_p=0.9,
                                pad_token_id=tokenizer.eos_token_id)
    
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

def generate_with_steering(model, tokenizer, prompt, steering_vector, scale=3.0):
    """Generate with steering vector injected at layer 6"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]
    
    def steering_hook(module, input, output):
        # Inject steering at layer 6
        hidden_states = output[0] if isinstance(output, tuple) else output
        seq_len = hidden_states.shape[1]
        if steering_vector.shape[1] <= seq_len:
            positions = min(steering_vector.shape[1], seq_len)
            hidden_states[:, -positions:, :] += scale * steering_vector[:, -positions:, :]

        return (hidden_states, *output[1:])
    
    handle = model.model.layers[6].register_forward_hook(steering_hook)
    
    try:
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200,
                                    do_sample=True, temperature=0.3, top_p=0.9,
                                    pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    finally:
        handle.remove()

if __name__ == "__main__":
    run_all_tests()