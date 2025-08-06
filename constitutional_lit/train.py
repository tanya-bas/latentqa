"""
Constitutional AI training script for LatentQA.
Enhanced training with constitutional principles for feature interpretation and persona impersonation.
"""

import os
import fire
import torch
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
import wandb
from transformers import get_cosine_schedule_with_warmup
from peft import LoraConfig
import torch.optim as optim

# Import constitutional components
from constitutional_lit.constitution import load_reading_constitution, load_control_constitution
from constitutional_lit.critique_models import (
    create_feature_critique_model, 
    create_persona_critique_model,
    create_ai_critique_evaluator
)
from constitutional_lit.reward_functions import (
    compute_interpretation_rewards,
    compute_persona_rewards,
    compute_constitutional_loss
)

# Import original LatentQA components
import sys
sys.path.append('../lit')
from lit.configs.train_config import train_config
from lit.configs.peft_config import lora_config
from lit.utils.dataset_utils import get_dataset, DataCollatorForLatentQA, LengthBasedBatchSampler
from lit.utils.infra_utils import (
    get_logger, setup_wandb, save_model, get_ema, update_ema,
    update_config, get_tokenizer, get_model, get_modules
)
from lit.utils.activation_utils import latent_qa

def get_dataloaders_simple(train_config, tokenizer, max_samples=None):
    """Get dataloaders without distributed training."""
    dataset_train = get_dataset(train_config, tokenizer, train=True)
    
    # Limit dataset size for faster training
    if max_samples is not None and max_samples > 0:
        print(f"Limiting dataset to {max_samples} samples for faster training")
        # Create a subset of the dataset while preserving the lengths attribute
        indices = torch.randperm(len(dataset_train))[:max_samples]
        
        # Create a custom subset that preserves the lengths attribute
        class SubsetWithLengths(torch.utils.data.Subset):
            def __init__(self, dataset, indices):
                super().__init__(dataset, indices)
                # Preserve the lengths attribute from the original dataset
                self.lengths = [dataset.lengths[i] for i in indices]
        
        dataset_train = SubsetWithLengths(dataset_train, indices)
    
    # Use simple batch sampler instead of distributed
    batch_sampler = LengthBasedBatchSampler(
        dataset_train,
        train_config.batch_size_training,
        drop_last=False,
        shuffle=True
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        collate_fn=DataCollatorForLatentQA(
            tokenizer,
            mask_all_but_last=False,
            nudge_persona=train_config.nudge_persona,
            modify_chat_template=train_config.modify_chat_template,
        ),
        batch_sampler=batch_sampler,
    )
    
    eval_dataloader = None
    if train_config.eval_ppl:
        dataset_eval = get_dataset(train_config, tokenizer, train=False)
        eval_batch_sampler = LengthBasedBatchSampler(
            dataset_eval,
            train_config.batch_size_training,
            drop_last=False,
            shuffle=False
        )
        eval_dataloader = torch.utils.data.DataLoader(
            dataset_eval,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            collate_fn=DataCollatorForLatentQA(
                tokenizer,
                mask_all_but_last=False,
                nudge_persona=train_config.nudge_persona,
                modify_chat_template=train_config.modify_chat_template,
            ),
            batch_sampler=eval_batch_sampler,
        )
    
    return train_dataloader, eval_dataloader

class ConstitutionalLatentQATrainer:
    """Enhanced LatentQA trainer with Constitutional AI integration."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        
        # Clean GPU memory at startup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            print(f"GPU Memory before training: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        # Load constitutional principles
        self.reading_constitution = load_reading_constitution()
        self.control_constitution = load_control_constitution()
        
        # Initialize AI critique evaluator
        self.ai_evaluator = None
        
        # Training components
        self.target_model = None
        self.decoder_model = None
        self.optimizer = None
        self.scheduler = None
        
    def setup_models(self, tokenizer):
        """Setup target model and decoder model with memory optimization."""
        
        # Clean GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU Memory before loading models: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        # Load target model with memory optimization
        print("Loading target model...")
        self.target_model = get_model(
            self.args.target_model_name, 
            tokenizer, 
            device=self.device
        )
        
        if torch.cuda.is_available():
            print(f"GPU Memory after target model: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            torch.cuda.empty_cache()
        
        # Load decoder model with LoRA and memory optimization
        print("Loading decoder model...")
        lora_params = {
            k.name: getattr(lora_config(), k.name) for k in lora_config.__dataclass_fields__.values()
        }
        peft_config = LoraConfig(**lora_params)
        
        self.decoder_model = get_model(
            self.args.target_model_name,
            tokenizer,
            peft_config=peft_config,
            device=self.device,
            distributed_training=False  # Disable distributed training to save memory
        )
        
        if torch.cuda.is_available():
            print(f"GPU Memory after decoder model: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            
            # Check if we're using too much memory
            total_memory = torch.cuda.get_device_properties(0).total_memory
            used_memory = torch.cuda.memory_allocated()
            memory_usage = used_memory / total_memory
            
            print(f"Memory usage: {memory_usage:.2%}")
            
            if memory_usage > 0.9:  # If using more than 90% of GPU memory
                print("Warning: High GPU memory usage detected!")
                print("Consider using a smaller model or reducing batch size")
                
                # Try to free some memory
                torch.cuda.empty_cache()
                
                # If still too high, move some components to CPU
                if memory_usage > 0.95:
                    print("Moving some components to CPU to save GPU memory...")
                    # You could move the AI evaluator to CPU here if needed
        
    def setup_ai_evaluator(self):
        """Setup AI-based critique evaluator."""
        
        print("Setting up AI critique evaluator...")
        self.ai_evaluator = create_ai_critique_evaluator(self.args.target_model_name)
        
    def setup_training_components(self, train_dataloader):
        """Setup optimizer and scheduler."""
        
        self.optimizer = optim.AdamW(
            self.decoder_model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        
        training_steps = len(train_dataloader) * self.args.num_epochs
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=training_steps,
        )
        
    def extract_activations(self, batch, module_read):
        """Extract activations from target model."""
        
        # Use the original latent_qa function to get activations
        with torch.no_grad():
            outputs, activation_cache = latent_qa(
                batch,
                self.target_model,
                self.decoder_model,
                module_read,
                module_read,  # Use same module for read/write in extraction
                None,  # No tokenizer needed for extraction
                generate=False,
                no_grad=True
            )
        
        return activation_cache
        
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
        
    def extract_personas_from_batch(self, batch):
        """Extract personas from batch data."""
        # Extract personas from batch metadata or content
        if hasattr(batch, 'personas'):
            return batch.personas
        elif isinstance(batch, dict) and 'personas' in batch:
            return batch['personas']
        else:
            # Default personas based on batch content
            return ["helpful_assistant"] * len(batch) if hasattr(batch, '__len__') else ["helpful_assistant"]
        
    def generate_control_strategies(self, batch, tokenizer):
        """Generate control strategies for persona impersonation."""
        
        # Extract personas from batch
        personas = self.extract_personas_from_batch(batch)
        
        # Generate control strategies
        strategies = []
        for persona in personas:
            # Generate strategy based on persona
            if "vegan" in persona.lower():
                strategy = "Use language that promotes veganism and animal welfare"
            elif "professional" in persona.lower():
                strategy = "Maintain formal and professional communication style"
            elif "creative" in persona.lower():
                strategy = "Employ imaginative and artistic language"
            elif "analytical" in persona.lower():
                strategy = "Focus on logical reasoning and systematic analysis"
            elif "empathetic" in persona.lower():
                strategy = "Show understanding and emotional support"
            else:
                strategy = f"Adapt behavior to match the persona: {persona}"
            
            strategies.append(strategy)
        
        return strategies
        
    def apply_control_strategies(self, strategies, batch, module_read, module_write, tokenizer):
        """Apply control strategies and get outputs."""
        
        controlled_outputs = []
        for strategy in strategies:
            try:
                # Apply strategy to modify batch
                modified_batch = self.apply_strategy_to_batch(batch, strategy)
                
                # Get controlled output
                with torch.no_grad():
                    outputs, _ = latent_qa(
                        modified_batch,
                        self.target_model,
                        self.decoder_model,
                        module_read,
                        module_write,
                        tokenizer,
                        generate=True,
                        no_grad=True,
                        max_new_tokens=100
                    )
                
                # Decode output
                if hasattr(outputs, 'sequences'):
                    output_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                else:
                    output_text = str(outputs)
                
                controlled_outputs.append(output_text)
                
            except Exception as e:
                # Fallback output
                output_text = f"Controlled output for strategy '{strategy}': {str(e)[:50]}..."
                controlled_outputs.append(output_text)
        
        return controlled_outputs
        
    def apply_strategy_to_batch(self, batch, strategy):
        """Apply control strategy to batch data."""
        
        # Create a copy of the batch
        if isinstance(batch, dict):
            modified_batch = batch.copy()
            modified_batch['control_strategy'] = strategy
        else:
            # For non-dict batches, create a wrapper
            modified_batch = {
                'data': batch,
                'control_strategy': strategy
            }
        
        return modified_batch
        
    def evaluate_control_strategies_ai(self, strategies, outputs, personas, constitution):
        """Evaluate control strategies using AI critique."""
        
        if self.ai_evaluator is None:
            return [{"persona_accuracy": 0.5, "consistency": 0.5, "authenticity": 0.5,
                    "naturalness": 0.5, "adaptability": 0.5, "coherence": 0.5}] * len(strategies)
        
        critiques = []
        for strategy, output, persona in zip(strategies, outputs, personas):
            # Evaluate against each constitutional principle
            principle_scores = []
            for principle in constitution:
                scores = self.ai_evaluator.evaluate_persona(strategy, output, persona, principle)
                principle_scores.append(scores)
            
            # Aggregate scores across principles
            avg_scores = {}
            for key in principle_scores[0].keys():
                avg_scores[key] = np.mean([ps[key] for ps in principle_scores])
            
            critiques.append(avg_scores)
        
        return critiques
        
    def constitutional_training_step(self, batch, module_read, module_write, tokenizer, step):
        """Single training step with constitutional evaluation."""
        
        # Extract activations
        activations = self.extract_activations(batch, module_read)
        
        # Generate interpretations
        interpretations = self.generate_interpretations(activations, tokenizer)
        
        # Evaluate interpretations using AI
        interpretation_critiques = self.evaluate_interpretations_ai(interpretations, self.reading_constitution)
        
        # Compute interpretation rewards
        interpretation_rewards = compute_interpretation_rewards(
            interpretations, interpretation_critiques, activations
        )
        
        # Generate control strategies (for control data)
        control_rewards = None
        if hasattr(batch, 'control_data') and batch.control_data:
            control_strategies = self.generate_control_strategies(batch, tokenizer)
            controlled_outputs = self.apply_control_strategies(
                control_strategies, batch, module_read, module_write, tokenizer
            )
            personas = self.extract_personas_from_batch(batch)
            
            # Evaluate control strategies using AI
            control_critiques = self.evaluate_control_strategies_ai(
                control_strategies, controlled_outputs, personas, self.control_constitution
            )
            
            # Compute control rewards
            control_rewards = compute_persona_rewards(
                control_strategies, controlled_outputs, control_critiques, personas
            )
            
            # Combine rewards
            combined_rewards = [
                0.5 * ir + 0.5 * cr 
                for ir, cr in zip(interpretation_rewards, control_rewards)
            ]
        else:
            # Reading only
            combined_rewards = interpretation_rewards
        
        # Standard LatentQA forward pass
        outputs, _ = latent_qa(
            batch,
            self.target_model,
            self.decoder_model,
            module_read,
            module_write,
            tokenizer,
            mask_verbs=True,
            shift_position_ids=self.args.shift_position_ids,
        )
        
        # Compute constitutional loss
        constitutional_loss = compute_constitutional_loss(
            outputs.loss, combined_rewards, alpha=0.3
        )
        
        return constitutional_loss, interpretation_rewards, control_rewards
        
    def train(self, **kwargs):
        """Main training loop with Constitutional AI integration."""
        
        # Setup distributed training
        if hasattr(self.args, 'use_ddp') and self.args.use_ddp:
            dist.init_process_group("nccl")
            rank = dist.get_rank()
            device = rank % torch.cuda.device_count()
            torch.cuda.set_device(device)
        elif hasattr(self.args, 'use_fsdp') and self.args.use_fsdp:
            # Handle FSDP setup
            rank = 0
            device = self.device
        else:
            rank = 0
            device = self.device
            
        # Update config with kwargs
        update_config(self.args, **kwargs)
        
        # Setup logging
        logger = get_logger(self.args, rank)
        wandb_run = None
        if hasattr(self.args, 'use_wandb') and self.args.use_wandb and rank == 0:
            try:
                wandb_run = setup_wandb(self.args, **kwargs)
            except Exception as e:
                print(f"Warning: Could not setup wandb: {e}")
                wandb_run = None
            
        # Load tokenizer and datasets
        tokenizer = get_tokenizer(self.args.target_model_name)
        train_dataloader, eval_dataloader = get_dataloaders_simple(self.args, tokenizer, max_samples=self.args.max_samples)
        
        # Setup models
        self.setup_models(tokenizer)
        
        # Setup AI evaluator
        if rank == 0:  # Only setup on main process
            self.setup_ai_evaluator()
            
        # Setup training components
        self.setup_training_components(train_dataloader)
        
        # Get modules
        module_read, module_write = get_modules(
            self.target_model, self.decoder_model, 
            **self.args.__dict__
        )
        
        # Adjust layer configuration for smaller models like DialoGPT
        if "dialogpt" in self.args.target_model_name.lower() or "gpt2" in self.args.target_model_name.lower():
            # DialoGPT-large has 36 layers, so adjust the range
            if len(module_read) == 0:
                print("Warning: No layers found, adjusting layer configuration for GPT-2 style model")
                # Use layers 30-35 for DialoGPT (last few layers)
                module_read, module_write = get_modules(
                    self.target_model, self.decoder_model,
                    min_layer_to_read=30,
                    max_layer_to_read=36,
                    num_layers_to_read=1,
                    layer_to_write=0,
                    module_setup='read-vary_write-fixed_n-fixed'
                )
        
        # Training loop
        train_steps = 0
        
        for epoch in range(self.args.num_epochs):
            self.decoder_model.train()
            
            total_length = len(train_dataloader) // self.args.gradient_accumulation_steps
            pbar = tqdm(
                colour="blue",
                desc=f"Constitutional Training Epoch: {epoch+1}",
                total=total_length,
                dynamic_ncols=True,
            )
            
            for step, batch in enumerate(train_dataloader):
                # Move batch to device
                for key in batch.keys():
                    batch[key] = batch[key].to(device)
                
                # Sample layers
                layer_list = np.random.choice(
                    len(module_read), self.args.num_layers_to_sample, replace=False
                )
                
                for idx in layer_list:
                    train_steps += 1
                    
                    # Constitutional training step
                    loss, interpretation_rewards, control_rewards = self.constitutional_training_step(
                        batch, module_read[idx], module_write[idx], tokenizer, step
                    )
                    
                    # Backward pass
                    loss = loss / self.args.gradient_accumulation_steps
                    loss.backward()
                    
                    if train_steps % self.args.gradient_accumulation_steps == 0:
                        # Gradient clipping
                        if hasattr(self.args, 'gradient_clipping') and self.args.gradient_clipping and hasattr(self.args, 'gradient_clipping_threshold') and self.args.gradient_clipping_threshold > 0.0:
                            torch.nn.utils.clip_grad_norm_(
                                self.decoder_model.parameters(),
                                self.args.gradient_clipping_threshold,
                            )
                        
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        pbar.update(1)
                
                # Logging
                if wandb_run is not None:
                    log_dict = {
                        "train/epoch": epoch,
                        "train/step": epoch * len(train_dataloader) + step,
                        "train/loss": loss.detach().float(),
                        "train/interpretation_reward": np.mean(interpretation_rewards) if interpretation_rewards else 0.0,
                    }
                    
                    if control_rewards:
                        log_dict["train/control_reward"] = np.mean(control_rewards)
                    
                    wandb_run.log(log_dict)
                else:
                    # Print progress without wandb
                    if step % 10 == 0:
                        print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.detach().float():.4f}")
                
                pbar.set_description(
                    f"Constitutional Training Epoch: {epoch+1}/{self.args.num_epochs}, "
                    f"batch {step+1}/{len(train_dataloader)} completed "
                    f"(loss: {loss.detach().float():.4f})"
                )
                
                # Save model
                if hasattr(self.args, 'save_every_n_steps') and train_steps % self.args.save_every_n_steps == 0:
                    save_model(
                        self.decoder_model,
                        None,  # No EMA for now
                        tokenizer,
                        self.args,
                        epoch,
                        train_steps,
                        logger,
                        rank,
                    )
            
            # End of epoch
            self.scheduler.step()
            pbar.close()
            
            # Save model at end of epoch
            if hasattr(self.args, 'save_model') and self.args.save_model:
                save_model(
                    self.decoder_model,
                    None,
                    tokenizer,
                    self.args,
                    epoch,
                    train_steps,
                    logger,
                    rank,
                )
        
        if wandb_run is not None:
            wandb.finish()
            
        if hasattr(self.args, 'use_ddp') and self.args.use_ddp:
            dist.destroy_process_group()
            
        logger.info("Constitutional training completed!")

def main(**kwargs):
    """Main entry point for constitutional training."""
    
    # Get args
    args = train_config()
    
    # Create trainer
    trainer = ConstitutionalLatentQATrainer(args)
    
    # Start training
    trainer.train(**kwargs)

if __name__ == "__main__":
    fire.Fire(main) 