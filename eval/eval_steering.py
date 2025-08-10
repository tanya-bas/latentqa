import os
import sys
import pandas as pd
import ast
import re
import torch
import random
import numpy as np

# Ensure project root is on sys.path when running via absolute path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lit.utils.infra_utils import get_model, get_tokenizer

from prompt import STEERING_PROMPT
from dotenv import load_dotenv
from eval_prompt import get_answer_llama

from activations_steering import extract_steering_vector, generate_with_steering



load_dotenv()



def parse_llama_answer_for_pairs(text):
    pairs = []
    # Try to parse as a Python literal first (in case the model returns a list of dicts)
    try:
        obj = ast.literal_eval(text)
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict) and "pos" in item and "neg" in item:
                    pairs.append({
                        "pos": str(item["pos"]).strip(),
                        "neg": str(item["neg"]).strip(),
                    })
        if pairs:
            return pairs
    except Exception:
        pass

    # Fallback: extract from <example_pair>...</example_pair> blocks
    blocks = re.findall(r"<example_pair>(.*?)</example_pair>", text, flags=re.DOTALL | re.IGNORECASE)
    for block in blocks:
        pos_match = re.search(r'"pos"\s*:\s*"(.*?)"', block, flags=re.DOTALL | re.IGNORECASE) or \
                    re.search(r"pos\s*:\s*\"(.*?)\"", block, flags=re.DOTALL | re.IGNORECASE)
        neg_match = re.search(r'"neg"\s*:\s*"(.*?)"', block, flags=re.DOTALL | re.IGNORECASE) or \
                    re.search(r"neg\s*:\s*\"(.*?)\"", block, flags=re.DOTALL | re.IGNORECASE)
        if pos_match and neg_match:
            pairs.append({
                "pos": pos_match.group(1).strip(),
                "neg": neg_match.group(1).strip(),
            })

    # Last resort: grab any inline pairs like "pos": "...", "neg": "..."
    if not pairs:
        inline_pairs = re.findall(r'"pos"\s*:\s*"(.*?)"\s*,\s*"neg"\s*:\s*"(.*?)"', text, flags=re.DOTALL | re.IGNORECASE)
        for pos, neg in inline_pairs:
            pairs.append({
                "pos": pos.strip(),
                "neg": neg.strip(),
            })

    return pairs



if __name__ == "__main__":

    df = pd.read_csv("results_control.csv")
    behaviour = df['Behaviour'][0]
    prompt = df['The prompt '][0]


    llama_answer = get_answer_llama(system_prompt = STEERING_PROMPT.format(TRAIT=behaviour, trait_instruction=prompt), user_prompt = '')
    example_pairs = parse_llama_answer_for_pairs(llama_answer)
    print(example_pairs)

    target_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    test_prompt = "What should I get for lunch? Be short and concise."
    
    tokenizer = get_tokenizer(target_model_name)
    model = get_model(target_model_name, tokenizer, device="cuda")

    vegan_vector = extract_steering_vector(model, tokenizer, example_pairs[0]['pos'])
    non_vegan_vector = extract_steering_vector(model, tokenizer, example_pairs[0]['neg'])
    
    min_len = min(vegan_vector.shape[1], non_vegan_vector.shape[1])
    contrast_vector = vegan_vector[:, :min_len, :] - non_vegan_vector[:, :min_len, :]

    print(generate_with_steering(model, tokenizer, test_prompt, contrast_vector))


    # Uncomment to inspect the raw model output
    # print(llama_answer)

    # print(STEERING_PROMPT)