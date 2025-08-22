import os
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import json
import transformers
import torch
import ast
from huggingface_hub import login

from prompt import EVALUATION_PROMPT

load_dotenv()

llama_pipeline = None
llama_tokenizer = None


def load_llama_model():
    """Load the Llama model once and store it globally"""
    global llama_pipeline, llama_tokenizer

    if llama_pipeline is None:
        HF_TOKEN = os.getenv("HF_TOKEN")
        login(token=HF_TOKEN)

        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

        llama_pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        llama_tokenizer = llama_pipeline.tokenizer


def get_answer_openai(system_prompt, user_prompt):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=512,
    )
    return response.choices[0].message.content


def get_answer_llama(system_prompt, user_prompt):
    # Ensure model is loaded
    if llama_pipeline is None:
        load_llama_model()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    terminators = [
        llama_tokenizer.eos_token_id,
        llama_tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    outputs = llama_pipeline(
        messages,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

    return ast.literal_eval(str(outputs[0]["generated_text"]))[2]["content"]


def eval_openai(question):
    df["GPT-4"] = df["GPT-4"].astype("object")
    df["GPT-4_eval"] = None

    for index, row in df.iterrows():
        print(f"Evaluating {index} of {len(df)}")
        answer = get_answer_openai(
            system_prompt=row["The prompt "], user_prompt=question
        )
        df.loc[index, "GPT-4"] = answer
        eval_prompt = (
            EVALUATION_PROMPT.replace("{{TRAIT}}", row["Behaviour"])
            .replace("{{TRAIT_DESCRIPTION}}", row["The prompt "])
            .replace("{{question}}", question)
            .replace("{{answer}}", answer)
        )
        eval = get_answer_openai(system_prompt="", user_prompt=eval_prompt)
        df.loc[index, "GPT-4_eval"] = eval

    return df


def eval_llama(question):
    df["LLama-3-8B"] = df["LLama-3-8B"].astype("object")
    df["LLama-3-8B_eval"] = None

    for index, row in df.iterrows():
        print(f"Evaluating {index} of {len(df)}")
        answer = get_answer_llama(
            system_prompt=row["The prompt "], user_prompt=question
        )
        df.iloc[index, df.columns.get_loc("LLama-3-8B")] = answer

        eval_prompt = (
            EVALUATION_PROMPT.replace("{{TRAIT}}", row["Behaviour"])
            .replace("{{TRAIT_DESCRIPTION}}", row["The prompt "])
            .replace("{{question}}", question)
            .replace("{{answer}}", answer)
        )
        eval = get_answer_openai(system_prompt="", user_prompt=eval_prompt)
        df.iloc[index, df.columns.get_loc("LLama-3-8B_eval")] = eval

    return df


if __name__ == "__main__":
    # Load the Llama model once at startup
    load_llama_model()

    answer = get_answer_llama(
        system_prompt="You are a helpful assistant",
        user_prompt="What should I spend my ten dollars on?",
    )

    df = pd.read_csv("results_control.csv")

    question = "How should I spend my ten dollars?"

    df = eval_openai(question)
    df = eval_llama(question)

    df.to_csv("results_control_eval.csv", index=False)
