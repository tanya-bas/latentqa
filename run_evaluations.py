#!/usr/bin/env python3
"""
Script to run evaluation prompts on all completion files in lora_completions and sft_completions directories.
"""

import json
import os
import glob
import time
import openai
from pathlib import Path

# OpenAI client configuration
openai_client = openai.OpenAI(
    base_url="https://syi8httobjtsxq-8000.proxy.runpod.net/v1", api_key="EMPTY"
)

# Import the evaluation prompt from eval/prompt.py
from eval.prompt import EVALUATION_PROMPT

# Model configuration
MODEL_NAME = "openai/gpt-oss-20b"


def find_trait_name(folder_name):
    """Extract trait name from folder name like 'ada-lovelace_dolly_samples30'"""
    return folder_name.replace("_dolly_samples30", "")


def load_questions(trait_name):
    """Load questions for a given trait from prompts directory"""
    prompt_file = f"prompts/{trait_name}.json"
    if os.path.exists(prompt_file):
        with open(prompt_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_responses(completion_file):
    """Load responses from a completion file"""
    with open(completion_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_trait_description(trait_name):
    """Load trait description from trait_data_eval directory"""
    trait_eval_file = f"trait_data_eval/{trait_name}.json"
    if os.path.exists(trait_eval_file):
        with open(trait_eval_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Get the first "pos" instruction
            if "instruction" in data and len(data["instruction"]) > 0:
                return data["instruction"][0]["pos"]

    # Fallback description if file doesn't exist or has unexpected structure
    return f"Embodying the {trait_name} trait"


def create_evaluation_prompt(trait, question, answer):
    """Create an evaluation prompt using the template"""
    trait_description = load_trait_description(trait)

    return (
        EVALUATION_PROMPT.replace("{{TRAIT}}", trait)
        .replace("{{TRAIT_DESCRIPTION}}", trait_description)
        .replace("{{question}}", question)
        .replace("{{answer}}", answer)
    )


def call_openai_api(prompt, max_retries=3, delay=1):
    """Call the OpenAI API with the evaluation prompt using the OpenAI SDK"""

    for attempt in range(max_retries):
        try:
            response = openai_client.completions.create(
                model=MODEL_NAME,
                prompt=prompt,
                max_tokens=2000,  # Increased to allow for complete responses
                temperature=0.0,  # We want consistent evaluation scores
            )

            if response.choices and len(response.choices) > 0:
                choice = response.choices[0]
                content = choice.text
                if content is not None:
                    return content.strip()
                else:
                    print(
                        f"    Warning: API returned None content. Choice finish_reason: {getattr(choice, 'finish_reason', 'unknown')}"
                    )
                    return "ERROR: None content in response"
            else:
                print(f"    Warning: No choices in API response. Response: {response}")
                return "ERROR: No choices in response"

        except Exception as e:
            print(f"    API call exception (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(delay * (attempt + 1))
                continue
            return f"ERROR: API call failed after {max_retries} attempts"

    return "ERROR: Max retries exceeded"


def process_completions_directory(directory_path):
    """Process all completion files in a directory"""
    results = {}

    # Get all trait directories
    trait_dirs = [
        d
        for d in os.listdir(directory_path)
        if os.path.isdir(os.path.join(directory_path, d))
    ]

    print(f"\nProcessing {directory_path}")
    print(f"Found {len(trait_dirs)} trait directories")

    for trait_dir in sorted(trait_dirs):
        trait_name = find_trait_name(trait_dir)
        trait_path = os.path.join(directory_path, trait_dir)

        # Find the JSON file in the trait directory
        json_files = glob.glob(os.path.join(trait_path, "*.json"))
        if not json_files:
            print(f"  No JSON files found in {trait_dir}")
            continue

        completion_file = json_files[0]

        # Load questions and responses
        questions = load_questions(trait_name)
        if questions is None:
            print(f"  No questions found for trait: {trait_name}")
            continue

        responses = load_responses(completion_file)

        if len(questions) != len(responses):
            print(
                f"  WARNING: Mismatch in {trait_name} - {len(questions)} questions vs {len(responses)} responses"
            )

        # Create evaluation prompts for each question-response pair and get scores
        trait_results = []
        print(f"  Processing {len(questions)} evaluations for {trait_name}...")

        for i, (question, response) in enumerate(zip(questions, responses)):
            eval_prompt = create_evaluation_prompt(trait_name, question, response)

            # Debug: Print first evaluation prompt to check formatting
            if i == 0:
                print(f"  Sample evaluation prompt (first 500 chars):")
                print(f"  {eval_prompt[:500]}...")
                print()

            # Call the API to get the evaluation score
            print(
                f"    Evaluating response {i+1}/{len(questions)}...",
                end=" ",
                flush=True,
            )
            score = call_openai_api(eval_prompt)
            print(f"Score: {score}")

            trait_results.append(
                {
                    "question_index": i,
                    "question": question,
                    "response": response,
                    "evaluation_prompt": eval_prompt,
                    "evaluation_score": score,
                }
            )

            # Small delay to avoid overwhelming the API
            time.sleep(0.5)

        results[trait_name] = trait_results
        print(f"  ✓ Completed {trait_name}: {len(trait_results)} evaluations")

    return results


def main():
    """Main function to process both directories"""
    print("=" * 80)
    print("EVALUATING COMPLETION FILES")
    print("=" * 80)

    # Process both directories
    lora_results = process_completions_directory("lora_completions")
    sft_results = process_completions_directory("sft_completions")

    # Save results
    output_data = {"lora_completions": lora_results, "sft_completions": sft_results}

    output_file = "evaluation_results_with_scores.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"LoRA Completions: {len(lora_results)} traits processed")
    print(f"SFT Completions: {len(sft_results)} traits processed")

    total_evaluations = sum(
        len(trait_data) for trait_data in lora_results.values()
    ) + sum(len(trait_data) for trait_data in sft_results.values())

    print(f"Total evaluations completed: {total_evaluations}")
    print(f"Results saved to: {output_file}")

    # Calculate and show score statistics
    def calculate_scores_stats(results, dataset_name):
        print(f"\n{dataset_name} Score Statistics:")
        print("-" * 40)

        for trait_name, trait_data in results.items():
            scores = []
            errors = 0

            for evaluation in trait_data:
                score_str = evaluation["evaluation_score"]
                try:
                    if score_str.startswith("ERROR"):
                        errors += 1
                    elif score_str == "REFUSAL":
                        # Count refusals separately
                        pass
                    else:
                        # Try to extract numeric score
                        score = float(score_str)
                        scores.append(score)
                except (ValueError, TypeError):
                    errors += 1

            if scores:
                avg_score = sum(scores) / len(scores)
                print(
                    f"  {trait_name}: Avg={avg_score:.1f} ({len(scores)} valid, {errors} errors)"
                )
            else:
                print(f"  {trait_name}: No valid scores ({errors} errors)")

    if lora_results:
        calculate_scores_stats(lora_results, "LoRA")

    if sft_results:
        calculate_scores_stats(sft_results, "SFT")

    # Show some examples
    print("\n" + "=" * 80)
    print("EXAMPLE EVALUATIONS")
    print("=" * 80)

    if lora_results:
        first_trait = list(lora_results.keys())[0]
        first_eval = lora_results[first_trait][0]
        print(f"\nExample from LoRA - {first_trait}:")
        print("-" * 40)
        print(f"Question: {first_eval['question']}")
        print(
            f"Response: {first_eval['response'][:150]}..."
            if len(first_eval["response"]) > 150
            else first_eval["response"]
        )
        print(f"Evaluation Score: {first_eval['evaluation_score']}")


if __name__ == "__main__":
    main()
