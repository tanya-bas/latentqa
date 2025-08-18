import argparse
import json
import os
import sys
import traceback
from glob import glob

# Ensure module execution from repo root: python -m latentqa.run_trait_controls ...


def load_trait_file(path):
    with open(path, "r") as f:
        data = json.load(f)
    # Handle both "instruction" and "instructions" keys gracefully
    instructions = data.get("instruction") or data.get("instructions") or []
    if not isinstance(instructions, list) or not instructions:
        raise ValueError(f"No instruction array found in {path}")
    first = instructions[0]
    if isinstance(first, dict):
        pos_prompt = first.get("pos")
    elif isinstance(first, str):
        pos_prompt = first
    else:
        pos_prompt = None
    if not pos_prompt or not isinstance(pos_prompt, str):
        raise ValueError(f"First instruction missing 'pos' string in {path}")
    questions = data.get("questions") or []
    if not isinstance(questions, list):
        raise ValueError(f"'questions' must be a list in {path}")
    return pos_prompt, questions


def write_prompts_file(prompts_dir, name, questions, overwrite=False):
    os.makedirs(prompts_dir, exist_ok=True)
    out_path = os.path.join(prompts_dir, f"{name}.json")
    if os.path.exists(out_path) and not overwrite:
        return out_path
    with open(out_path, "w") as f:
        json.dump(questions, f, indent=2)
        print('writing to ', out_path, 'with questions', questions)
    return out_path


def run_reading(control_name, prompt, target_model_name, decoder_model_name, min_layer_to_read=None, max_layer_to_read=None):
    # Import here to avoid import cost if just listing
    from lit import reading as lit_reading

    kwargs = {
        "target_model_name": target_model_name,
        "decoder_model_name": decoder_model_name,
        "prompt": prompt,
        "save_name": control_name,
    }
    # Optional overrides
    if min_layer_to_read is not None:
        kwargs["min_layer_to_read"] = int(min_layer_to_read)
    if max_layer_to_read is not None:
        kwargs["max_layer_to_read"] = int(max_layer_to_read)

    # This will save controls/<control_name>.json
    lit_reading.main(**kwargs)


def run_control(control_name, target_model_name, decoder_model_name, dataset, samples, eval_prompts, per_layer_loss=True, save_model=False):
    from lit import control as lit_control

    kwargs = {
        "target_model_name": target_model_name,
        "decoder_model_name": decoder_model_name,
        "control": control_name,
        "dataset": dataset,
        "samples": int(samples),
        "eval_prompts": eval_prompts,
        "per_layer_loss": bool(per_layer_loss),
        "save_model": bool(save_model),
    }
    lit_control.main(**kwargs)


def main():
    parser = argparse.ArgumentParser(description="Generate controls and run steering for all traits.")
    parser.add_argument("--trait_dir", type=str, default="latentqa/trait_data_extract", help="Directory containing trait JSON files")
    parser.add_argument("--prompts_dir", type=str, default="prompts", help="Directory to write eval prompt files")
    parser.add_argument("--target_model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Target model identifier")
    parser.add_argument("--decoder_model_name", type=str, default="aypan17/latentqa_llama-3-8b-instruct", help="Decoder (PEFT) model identifier or path")
    parser.add_argument("--dataset", type=str, default="dolly", choices=["dolly", "alpaca"], help="Dataset for steering stage")
    parser.add_argument("--samples", type=int, default=30, help="Number of samples for steering stage")
    parser.add_argument("--per_layer_loss", action="store_true", help="Use per-layer loss variant (slower)")
    parser.add_argument("--save_model", action="store_true", help="Save the steered model after training")
    parser.add_argument("--overwrite_prompts", action="store_true", help="Overwrite prompts files if they already exist")
    parser.add_argument("--only_list", action="store_true", help="List planned runs without executing")
    # parser.add_argument("--min_layer_to_read", type=int, default=None, help="Optional override for reading stage")
    # parser.add_argument("--max_layer_to_read", type=int, default=None, help="Optional override for reading stage")

    args = parser.parse_args()

    # Resolve absolute paths for robustness
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    trait_dir = args.trait_dir
    if not os.path.isabs(trait_dir):
        trait_dir = os.path.abspath(os.path.join(repo_root, trait_dir))
    prompts_dir = args.prompts_dir
    if not os.path.isabs(prompts_dir):
        prompts_dir = os.path.abspath(os.path.join(repo_root, prompts_dir))

    trait_paths = sorted(glob(os.path.join(trait_dir, "*.json")))
    if not trait_paths:
        print(f"No trait JSON files found in {trait_dir}")
        sys.exit(1)

    planned = []
    for path in trait_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            prompt, questions = load_trait_file(path)
        except Exception as e:
            print(f"Skipping {path}: {e}")
            continue
        planned.append((name, prompt, questions))

    print(f"Found {len(planned)} trait files.")
    if args.only_list:
        for name, _, questions in planned:
            print(f"- {name}: {len(questions)} eval questions")
        return

    print('questions', questions)
    for idx, (name, prompt, questions) in enumerate(planned, start=1):
        print(f"[{idx}/{len(planned)}] Processing trait '{name}' with {len(questions)} questions...")
        try:
            # 1) Create prompts/<name>.json for control eval
            eval_path = write_prompts_file(prompts_dir, name, questions, overwrite=args.overwrite_prompts)
            print(f"  - Wrote eval prompts: {eval_path}")

            # 2) Run reading to generate controls/<name>.json (QA pairs)
            print(f"  - Generating QA pairs via reading for control '{name}'...")
            run_reading(
                control_name=name,
                prompt=prompt,
                target_model_name=args.target_model_name,
                decoder_model_name=args.decoder_model_name,
            )

            # 3) Run control with eval prompts to get completions
            print(f"  - Running control and generating completions for '{name}'...")
            run_control(
                control_name=name,
                target_model_name=args.target_model_name,
                decoder_model_name=args.decoder_model_name,
                dataset=args.dataset,
                samples=args.samples,
                eval_prompts=name,
                per_layer_loss=args.per_layer_loss,
                save_model=args.save_model,
            )
        except Exception:
            print(f"  ! Error while processing '{name}':")
            traceback.print_exc()


if __name__ == "__main__":
    main()


