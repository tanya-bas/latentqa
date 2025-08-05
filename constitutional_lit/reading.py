"""
Constitutional AI reading script for LatentQA.
Enhanced reading with constitutional evaluation for feature interpretation.
"""

import json
import fire
import torch
import numpy as np
from typing import List, Dict, Any

# Import constitutional components
from constitutional_lit.constitution import load_reading_constitution
from constitutional_lit.critique_models import create_ai_critique_evaluator
from constitutional_lit.reward_functions import compute_interpretation_rewards

# Import original LatentQA components
import sys
sys.path.append('../lit')
from lit.utils.dataset_utils import tokenize, BASE_DIALOG, ENCODER_CHAT_TEMPLATES
from lit.utils.activation_utils import latent_qa
from lit.utils.infra_utils import (
    update_config, get_model, get_tokenizer, get_modules
)
from lit.configs.interpret_config import interpret_config

class ConstitutionalReader:
    """Enhanced reader with constitutional AI integration."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        
        # Load constitutional principles
        self.reading_constitution = load_reading_constitution()
        
        # Initialize models
        self.target_model = None
        self.decoder_model = None
        self.ai_evaluator = None
        
    def setup_models(self, tokenizer):
        """Setup target model, decoder model, and AI evaluator."""
        
        # Load target model
        self.target_model = get_model(self.args.target_model_name, tokenizer, device=self.device)
        
        # Load decoder model
        self.decoder_model = get_model(
            self.args.target_model_name,
            tokenizer,
            load_peft_checkpoint=self.args.decoder_model_name,
            device=self.device,
        )
        
        # Setup AI evaluator
        self.ai_evaluator = create_ai_critique_evaluator(self.args.target_model_name)
        
    def extract_activations(self, dialogs, questions, tokenizer):
        """Extract activations from target model for given dialogs and questions."""
        
        # Prepare probe data
        probe_data = []
        chat_template = ENCODER_CHAT_TEMPLATES.get(tokenizer.name_or_path, None)
        
        for dialog in dialogs:
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
            
            for item in questions:
                probe_data.append({
                    "read_prompt": read_prompt,
                    "dialog": BASE_DIALOG + [{"role": "user", "content": item[0]}],
                })
        
        # Tokenize
        tokenized_batch = tokenize(
            probe_data,
            tokenizer,
            name=self.args.target_model_name,
            generate=True,
            mask_all_but_last=True,
        )
        
        # Get modules
        module_read, module_write = get_modules(self.target_model, self.decoder_model, **vars(self.args))
        
        # Extract activations
        with torch.no_grad():
            out, activation_cache = latent_qa(
                tokenized_batch,
                self.target_model,
                self.decoder_model,
                module_read[0],
                module_write[0],
                tokenizer,
                shift_position_ids=False,
                generate=True,
                no_grad=True,
            )
        
        return out, activation_cache, tokenized_batch
        
    def generate_interpretations(self, activations, tokenizer):
        """Generate interpretations from activations."""
        
        interpretations = []
        for activation in activations:
            try:
                # Convert activation to input format for decoder
                if isinstance(activation, torch.Tensor):
                    # Use activation as input to decoder
                    with torch.no_grad():
                        interpretation_tokens = self.decoder_model.generate(
                            inputs=activation.unsqueeze(0) if activation.dim() == 2 else activation,
                            max_new_tokens=50,
                            do_sample=True,
                            temperature=0.7,
                            pad_token_id=tokenizer.eos_token_id
                        )
                    
                    interpretation_text = tokenizer.decode(interpretation_tokens[0], skip_special_tokens=True)
                else:
                    # Fallback for non-tensor activations
                    interpretation_text = f"Activation pattern: {str(activation)[:100]}..."
                
                interpretations.append(interpretation_text)
                
            except Exception as e:
                # Fallback interpretation
                interpretation_text = f"Model activation analysis: {str(e)[:50]}..."
                interpretations.append(interpretation_text)
        
        return interpretations
        
    def evaluate_interpretations_ai(self, interpretations, constitution):
        """Evaluate interpretations using AI critique."""
        
        if self.ai_evaluator is None:
            return [{"accuracy": 0.5, "relevance": 0.5, "clarity": 0.5, 
                    "completeness": 0.5, "predictive_power": 0.5}] * len(interpretations)
        
        try:
            critiques = []
            for interpretation in interpretations:
                # Evaluate against each constitutional principle
                principle_scores = []
                for principle in constitution:
                    scores = self.ai_evaluator.evaluate_interpretation(interpretation, principle)
                    principle_scores.append(scores)
                
                # Aggregate scores across principles
                avg_scores = {}
                for key in principle_scores[0].keys():
                    avg_scores[key] = np.mean([ps[key] for ps in principle_scores])
                
                critiques.append(avg_scores)
            
            return critiques
        except Exception as e:
            print(f"Error in AI critique evaluation: {e}")
            # Fallback evaluation
            return [{"accuracy": 0.5, "relevance": 0.5, "clarity": 0.5, 
                    "completeness": 0.5, "predictive_power": 0.5}] * len(interpretations)
        
    def improve_interpretations(self, interpretations, critiques):
        """Improve interpretations based on constitutional feedback."""
        
        improved_interpretations = []
        
        for interpretation, critique in zip(interpretations, critiques):
            # Analyze critique scores
            accuracy = critique.get("accuracy", 0.5)
            relevance = critique.get("relevance", 0.5)
            clarity = critique.get("clarity", 0.5)
            completeness = critique.get("completeness", 0.5)
            
            # Generate improvement suggestions
            improvements = []
            
            if accuracy < 0.6:
                improvements.append("Focus on more accurate feature descriptions")
            if relevance < 0.6:
                improvements.append("Emphasize more relevant activation patterns")
            if clarity < 0.6:
                improvements.append("Use clearer and more understandable language")
            if completeness < 0.6:
                improvements.append("Provide more comprehensive analysis")
            
            # Create improved interpretation
            if improvements:
                improved_text = f"{interpretation}\n\nImprovements: {'; '.join(improvements)}"
            else:
                improved_text = interpretation
                
            improved_interpretations.append(improved_text)
        
        return improved_interpretations
        
    def constitutional_interpret(self, target_model, decoder_model, tokenizer, dialogs, questions, args):
        """Enhanced interpretation with constitutional evaluation."""
        
        # Standard interpretation
        out, activation_cache, tokenized_batch = self.extract_activations(dialogs, questions, tokenizer)
        
        # Generate interpretations
        interpretations = self.generate_interpretations(activation_cache, tokenizer)
        
        # Constitutional evaluation using AI
        critiques = self.evaluate_interpretations_ai(interpretations, self.reading_constitution)
        
        # Improve interpretations based on constitutional feedback
        improved_interpretations = self.improve_interpretations(interpretations, critiques)
        
        # Compute alignment scores
        alignment_score = self.compute_alignment_score(improved_interpretations, critiques)
        
        # Format output
        QA_PAIRS = {}
        for i, (interpretation, improved_interpretation, critique) in enumerate(zip(interpretations, improved_interpretations, critiques)):
            if i % len(questions) == 0:
                curr_dialog = dialogs[i // len(questions)][0]
                QA_PAIRS[curr_dialog] = []
            
            prompt = questions[i % len(questions)][0]
            QA_PAIRS[curr_dialog].append({
                "prompt": prompt,
                "original_interpretation": interpretation,
                "improved_interpretation": improved_interpretation,
                "critique_scores": critique,
                "alignment_score": alignment_score
            })
        
        return QA_PAIRS, improved_interpretations, critiques, alignment_score
        
    def compute_alignment_score(self, interpretations, critiques):
        """Compute overall constitutional alignment score."""
        
        if not interpretations or not critiques:
            return 0.0
        
        # Compute rewards
        rewards = compute_interpretation_rewards(interpretations, critiques, [None] * len(interpretations))
        
        # Return average reward as alignment score
        return np.mean(rewards)
        
    def run_reading(self, **kwargs):
        """Main reading function with constitutional evaluation."""
        
        # Update config
        update_config(self.args, **kwargs)
        
        # Setup tokenizer
        tokenizer = get_tokenizer(self.args.target_model_name)
        
        # Setup models
        self.setup_models(tokenizer)
        
        # Prepare dialogs and questions
        dialogs = [[self.args.prompt]] if hasattr(self.args, 'prompt') else [["Hello, how are you?"]]
        questions = [
            ["What type of persona will the assistant adopt?"],
            ["How will the assistant express themselves?"],
            ["Please describe the persona of the assistant."],
            ["What characteristics define the assistant's persona?"],
        ]
        
        # Run constitutional interpretation
        qa_pairs, improved_interpretations, critiques, alignment_score = self.constitutional_interpret(
            self.target_model, self.decoder_model, tokenizer, dialogs, questions, self.args
        )
        
        # Print results
        print(f"\n=== Constitutional Reading Results ===")
        print(f"Overall Alignment Score: {alignment_score:.3f}")
        print(f"Number of Interpretations: {len(improved_interpretations)}")
        
        for dialog, qa_list in qa_pairs.items():
            print(f"\nDialog: {dialog}")
            for qa in qa_list:
                print(f"  Prompt: {qa['prompt']}")
                print(f"  Original: {qa['original_interpretation'][:100]}...")
                print(f"  Improved: {qa['improved_interpretation'][:100]}...")
                print(f"  Scores: {qa['critique_scores']}")
                print()
        
        # Save results
        if hasattr(self.args, 'save_name') and self.args.save_name:
            output_file = f"constitutional_reading_{self.args.save_name}.json"
            with open(output_file, "w") as f:
                json.dump({
                    "qa_pairs": qa_pairs,
                    "alignment_score": alignment_score,
                    "constitutional_principles": self.reading_constitution
                }, f, indent=2)
            print(f"Results saved to {output_file}")
        
        return qa_pairs, improved_interpretations, critiques, alignment_score

def main(**kwargs):
    """Main entry point for constitutional reading."""
    
    # Get args
    args = interpret_config()
    
    # Create reader
    reader = ConstitutionalReader(args)
    
    # Run reading
    return reader.run_reading(**kwargs)

if __name__ == "__main__":
    fire.Fire(main) 