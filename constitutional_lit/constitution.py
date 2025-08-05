"""
Constitutional principles for LatentQA training.
Defines the core principles for feature interpretation (reading) and persona impersonation (control).
"""

# Constitutional principles for reading/feature interpretation
READING_CONSTITUTION = [
    "Choose the description that most accurately captures the model's learned features.",
    "Choose the description that best explains the model's decision-making process.",
    "Choose the description that reveals the most relevant internal representations.",
    "Choose the description that helps understand the model's behavioral patterns.",
    "Choose the description that identifies the most important activation patterns.",
    "Choose the description that provides the clearest insight into model behavior.",
    "Choose the description that best translates neural activations to human concepts.",
    "Choose the description that enables the most effective model analysis."
]

# Constitutional principles for control/persona impersonation
CONTROL_CONSTITUTION = [
    "Choose the control that most effectively impersonates the target persona.",
    "Choose the control that maintains consistent persona behavior across contexts.",
    "Choose the control that captures the persona's core characteristics.",
    "Choose the control that enables natural persona expression.",
    "Choose the control that preserves persona authenticity.",
    "Choose the control that adapts persona appropriately to different situations.",
    "Choose the control that enables persona-driven decision making.",
    "Choose the control that maintains persona coherence throughout interactions."
]

def load_reading_constitution():
    """Load constitutional principles for reading operations."""
    return READING_CONSTITUTION

def load_control_constitution():
    """Load constitutional principles for control operations."""
    return CONTROL_CONSTITUTION

def load_constitutional_principles(operation_type):
    """Load constitutional principles based on operation type."""
    if operation_type == "reading" or operation_type == "interpretation":
        return load_reading_constitution()
    elif operation_type == "control" or operation_type == "persona":
        return load_control_constitution()
    else:
        raise ValueError(f"Unknown operation type: {operation_type}") 