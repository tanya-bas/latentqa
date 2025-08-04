#!/usr/bin/env python3
"""
Example script for training with SFT (Supervised Fine-Tuning) instead of LoRA.

Usage:
    # Single GPU SFT training
    python examples/train_sft.py
    
    # Multi-GPU SFT training  
    torchrun --nproc_per_node=2 examples/train_sft.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lit.train import main

if __name__ == "__main__":
    # Configure for SFT training
    main(
        peft_method="sft",           # Use supervised fine-tuning
        use_peft=False,              # Disable PEFT since we're doing full fine-tuning
        lr=1e-5,                     # Lower learning rate for full model training
        batch_size_training=2,       # Smaller batch size due to higher memory usage
        gradient_accumulation_steps=4, # More accumulation steps to maintain effective batch size
        num_epochs=1,                # Fewer epochs as full fine-tuning converges faster
        save_every_n_steps=1000,     # Save less frequently
        run_name="sft_training_example"
    )