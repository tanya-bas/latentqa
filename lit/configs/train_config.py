from dataclasses import dataclass
from typing import Optional


@dataclass
class train_config:
    # Model and dataset
    target_model_name: str = "microsoft/DialoGPT-medium"  # Publicly available model
    load_model_checkpoint: str = ""    
    train_system: str = ""
    train_stimulus_completion: str = ""
    train_stimulus: str = ""
    train_control: str = ""
    train_qa: str = ""
    train_with_verb_mask: Optional[str] = "user"
    add_thought_tokens: bool = False
    # Adds a nudge 'Follow the instructions' to make the model more faithful to the control
    nudge_persona: bool = False  
    modify_chat_template: bool = True
    filter: str = ""
    train_percent: float = 1.0

    # Evaluation and logging args
    eval_ppl: bool = False
    eval_system: str = ""
    eval_stimulus_completion: str = ""
    eval_stimulus: str = ""
    eval_control: str = ""
    eval_qa: str = ""
    eval_every_n_steps: int = 500
    # Please change to a directory with ample space as model checkpoints are saved here
    output_dir: str = "out/runs"
    save_model: bool = True
    save_every_n_steps: int = 500
    use_wandb: bool = False
    run_name: str = ""

    # Patching args
    shift_position_ids: bool = True
    min_layer_to_read: int = 15
    # Typically, we only read 1 layer, so set max = min + 1  
    max_layer_to_read: int = 16  
    layer_to_write: int = 0
    # Change only if reading from multiple layers (at once or sequentially) during training
    module_setup: str = "read-vary_write-fixed_n-fixed"
    num_layers_to_read: int = 1
    num_layers_to_sample: int = 1

    # Training args
    batch_size_training: int = 2  # Reduced batch size for memory
    gradient_accumulation_steps: int = 1
    gradient_clipping: bool = False
    gradient_clipping_threshold: float = 1.0
    num_epochs: int = 5
    num_workers_dataloader: int = 1
    lr: float = 3e-5
    ema_decay: float = 1
    warmup_steps: int = 0
    weight_decay: float = 0.01
    gamma: float = 0.85
    seed: int = 42
    # Set to 'None' for full model training
    peft_method: str = "lora"
    use_peft: bool = True
    use_fsdp: bool = False
    
    # Constitutional AI parameters
    constitutional_alpha: float = 0.3
    use_simple_critique: bool = False
    constitutional_evaluation: bool = True
    
    # Memory optimization parameters
    use_cpu_offload: bool = False
    max_memory_usage: float = 0.8  # Use 80% of GPU memory max
    