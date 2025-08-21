import json
import os
import traceback
from tempfile import NamedTemporaryFile
from pathlib import Path


# Ensure module execution from repo root: python -m latentqa.run_trait_controls ...
def load_trait_file(path):
    with open(path, "r") as f:
        obj = json.load(f)
        questions = obj.get("questions", [])
        prompt = obj.get("instruction", [])[0].get("pos", "")
        if not questions:
            raise ValueError(f"no questions in {path}")
        if not prompt:
            raise ValueError(f"no prompt in {path}")
        return prompt, questions


def write_prompts_file(prompts_dir, name, questions, overwrite=False):
    os.makedirs(prompts_dir, exist_ok=True)
    out_path = os.path.join(prompts_dir, f"{name}.json")
    with open(out_path, "w") as f:
        json.dump(questions, f, indent=2)
    return out_path


def run_reading(**kwargs):
    # Import here to avoid import cost if just listing
    from lit import reading as lit_reading

    # This will save controls/<control_name>.json
    lit_reading.main(**kwargs)


def run_control(**kwargs):
    from lit import control as lit_control

    lit_control.main(**kwargs)


def main():
    trait_dir = Path(__file__).resolve().parent.joinpath("trait_data_eval").resolve()
    print(f"Trait directory: {trait_dir}")

    target_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    decoder_model_name = "aypan17/latentqa_llama-3-8b-instruct"
    planned = []
    for path in trait_dir.glob("*.json"):
        name = path.stem
        prompt, questions = load_trait_file(path)
        planned.append((name, prompt, questions))

    print(planned)

    print(f"Found {len(planned)} traits.")

    for idx, (name, prompt, questions) in enumerate(planned, start=1):
        print(
            f"[{idx}/{len(planned)}] Processing trait '{name}' with {len(questions)} questions..."
        )
        try:
            # 1) Run reading to generate controls/<name>.json (QA pairs)
            print(f"  - Generating QA pairs via reading for control '{name}'...")
            reading_kwargs = {
                "target_model_name": target_model_name,
                "decoder_model_name": decoder_model_name,
                "prompt": prompt,
                "save_name": name,
            }
            run_reading(**reading_kwargs)

            eval_prompts = NamedTemporaryFile("w+b")
            eval_prompts.write(json.dumps(questions))
            kwargs = {
                "target_model_name": target_model_name,
                "decoder_model_name": decoder_model_name,
                "control": name,
                "samples": 30,
                "eval_prompts": eval_prompts.name,
                "per_layer_loss": True,
            }
            # 2) Run control with eval prompts to get completions
            print(f"  - Running control and generating completions for '{name}'...")
            run_control(**kwargs)
        except Exception:
            print(f"  ! Error while processing '{name}':")
            traceback.print_exc()


if __name__ == "__main__":
    main()
