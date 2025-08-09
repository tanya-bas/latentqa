import os
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import json
import transformers
import torch

from huggingface_hub import login

from prompt import EVALUATION_PROMPT

# Load environment variables from .env file
load_dotenv()

def get_answer_openai(system_prompt, user_prompt):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
        max_tokens=256
    )
    return response.choices[0].message.content

def get_answer_llama(system_prompt, user_prompt):
    HF_TOKEN = os.getenv("HF_TOKEN")
    login(token=HF_TOKEN)

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
    ]

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
    messages,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
    )

    return outputs[0]["generated_text"][-1]


if __name__ == "__main__":
    df = pd.read_csv("results_control.csv")
    df['GPT-4'] = df['GPT-4'].astype('object')
    df['GPT-4_eval'] = None

    for index, row in df.iterrows():
        question = "How should I spend my ten dollars?"
        answer = get_answer_openai(system_prompt = row["The prompt "], user_prompt = question)
        df.loc[index, 'GPT-4'] = answer 
        eval_prompt = EVALUATION_PROMPT.format(TRAIT=row["Behaviour"], TRAIT_DESCRIPTION=row["The prompt "], question=question, answer=answer)
        eval = get_answer_openai(system_prompt = '', user_prompt = eval_prompt)
        df.loc[index, 'GPT-4_eval'] = eval
        if index > 5:
            break

    df.to_csv("results_control_eval.csv", index=False)
