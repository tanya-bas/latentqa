# Constitutional AI for LatentQA

This directory contains the implementation of Constitutional AI principles for LatentQA, enhancing both reading (feature interpretation) and control (persona impersonation) operations with principled evaluation and improvement.

## Overview

Constitutional AI for LatentQA applies explicit principles to guide the training and evaluation of:
- **Reading Operations**: Feature interpretation with constitutional evaluation
- **Control Operations**: Persona impersonation with constitutional evaluation

## Architecture

```
constitutional_lit/
├── __init__.py                 # Module initialization
├── constitution.py             # Constitutional principles
├── critique_models.py          # AI critique models for evaluation
├── reward_functions.py         # Reward computation functions
├── train.py                   # Enhanced training with constitutional AI
├── reading.py                 # Enhanced reading with constitutional evaluation
├── control.py                 # Enhanced control with constitutional evaluation
└── README.md                  # This file
```

## Constitutional Principles

### Reading Principles (Feature Interpretation)
1. **Accuracy**: Choose descriptions that most accurately capture learned features
2. **Relevance**: Choose descriptions that reveal most relevant internal representations
3. **Clarity**: Choose descriptions that best translate neural activations to human concepts
4. **Completeness**: Choose descriptions that provide comprehensive analysis
5. **Predictive Power**: Choose descriptions that help predict model behavior

### Control Principles (Persona Impersonation)
1. **Persona Accuracy**: Choose controls that most effectively impersonate target personas
2. **Consistency**: Choose controls that maintain consistent persona behavior
3. **Authenticity**: Choose controls that preserve persona authenticity
4. **Naturalness**: Choose controls that enable natural persona expression
5. **Adaptability**: Choose controls that adapt persona appropriately to context
6. **Coherence**: Choose controls that maintain persona coherence throughout interactions

## Usage

### Training with Constitutional AI

```bash
# Run constitutional training
python -m constitutional_lit.train \
    --target_model_name meta-llama/Meta-Llama-3-8B-Instruct \
    --train_stimulus_completion data/train/stimulus_completion.json \
    --train_stimulus data/train/stimulus.json \
    --train_control data/train/control.json \
    --train_qa data/train/qa.json \
    --gradient_accumulation_steps 8 \
    --use_wandb
```

### Reading with Constitutional AI

```bash
# Run constitutional reading
python -m constitutional_lit.reading \
    --target_model_name meta-llama/Meta-Llama-3-8B-Instruct \
    --decoder_model_name path/to/decoder \
    --prompt "Hello, how are you?" \
    --save_name constitutional_reading_results
```

### Control with Constitutional AI

```bash
# Run constitutional control
python -m constitutional_lit.control \
    --target_model_name meta-llama/Meta-Llama-3-8B-Instruct \
    --decoder_model_name path/to/decoder \
    --control promote_veganism \
    --save_name constitutional_control_results
```

## Key Components

### 1. Constitutional Principles (`constitution.py`)
- Defines explicit principles for reading and control operations
- Provides loading functions for different operation types
- Ensures consistent evaluation criteria

### 2. Critique Models (`critique_models.py`)
- `FeatureInterpretationCritique`: Evaluates interpretation quality
- `PersonaImpersonationCritique`: Evaluates persona impersonation quality
- Uses transformer-based models with custom evaluation heads
- Provides fallback simple evaluation for testing

### 3. Reward Functions (`reward_functions.py`)
- `compute_interpretation_rewards`: Computes rewards for feature interpretation
- `compute_persona_rewards`: Computes rewards for persona impersonation
- `compute_constitutional_loss`: Combines base loss with constitutional rewards
- Normalization and alignment score computation

### 4. Enhanced Training (`train.py`)
- `ConstitutionalLatentQATrainer`: Main training class
- Integrates constitutional evaluation into training loop
- Supports both reading and control training
- Enhanced logging with constitutional metrics

### 5. Enhanced Reading (`reading.py`)
- `ConstitutionalReader`: Enhanced reading with constitutional evaluation
- Generates and evaluates interpretations
- Improves interpretations based on constitutional feedback
- Computes alignment scores

### 6. Enhanced Control (`control.py`)
- `ConstitutionalController`: Enhanced control with constitutional evaluation
- Generates and evaluates control strategies
- Improves strategies based on constitutional feedback
- Computes persona alignment scores

## Benefits

### 1. **Principled Evaluation**
- Explicit criteria for quality assessment
- Consistent evaluation across different scenarios
- Transparent decision-making process

### 2. **Improved Quality**
- Better feature interpretations through constitutional guidance
- More effective persona impersonation
- Continuous improvement through feedback

### 3. **Scalability**
- Automated evaluation reduces human annotation burden
- Consistent evaluation at scale
- Continuous improvement through AI feedback

### 4. **Transparency**
- Explicit constitutional principles
- Auditable evaluation process
- Clear improvement suggestions

## Example Output

### Reading Results
```json
{
  "qa_pairs": {
    "Hello, how are you?": [
      {
        "prompt": "What type of persona will the assistant adopt?",
        "original_interpretation": "The model is processing user input...",
        "improved_interpretation": "The model is processing user input with a friendly, helpful persona...",
        "critique_scores": {
          "accuracy": 0.8,
          "relevance": 0.7,
          "clarity": 0.9,
          "completeness": 0.6,
          "predictive_power": 0.7
        },
        "alignment_score": 0.74
      }
    ]
  },
  "alignment_score": 0.74,
  "constitutional_principles": [...]
}
```

### Control Results
```json
{
  "control_name": "promote_veganism",
  "strategies": [
    "Use language that promotes veganism and animal welfare\n\nImprovements: Strengthen persona alignment"
  ],
  "outputs": [
    "I'm passionate about promoting veganism and animal welfare..."
  ],
  "critiques": [
    {
      "persona_accuracy": 0.8,
      "consistency": 0.7,
      "authenticity": 0.9,
      "naturalness": 0.8,
      "adaptability": 0.6,
      "coherence": 0.8
    }
  ],
  "alignment_score": 0.77,
  "constitutional_principles": [...]
}
```

## Configuration

### Training Configuration
- `alpha`: Weight for constitutional component in loss (default: 0.3)
- `critique_model_path`: Path to pre-trained critique model
- `constitutional_evaluation`: Enable/disable constitutional evaluation

### Reading Configuration
- `prompt`: Input prompt for reading
- `save_name`: Name for saving results
- `critique_model_path`: Path to critique model

### Control Configuration
- `control`: Name of control file to use
- `save_name`: Name for saving results
- `critique_model_path`: Path to critique model

## Integration with Original LatentQA

The constitutional implementation is designed to be a drop-in enhancement to the original LatentQA:

1. **Backward Compatibility**: Original LatentQA scripts still work
2. **Optional Enhancement**: Constitutional features can be enabled/disabled
3. **Same Interface**: Uses same command-line interface with additional options
4. **Enhanced Output**: Provides additional constitutional evaluation metrics

## Future Enhancements

1. **Dynamic Constitutions**: Constitutions that adapt based on context
2. **Multi-Agent Constitutional AI**: Multiple agents following shared principles
3. **Human-AI Constitutional Collaboration**: Human oversight of constitutional decisions
4. **Domain-Specific Constitutions**: Specialized principles for different domains

## Citation

If you use this constitutional AI implementation, please cite:
