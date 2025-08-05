"""
Reward functions for Constitutional AI in LatentQA.
Computes rewards based on constitutional alignment for feature interpretation and persona impersonation.
"""

import torch
import numpy as np
from typing import List, Dict, Any

def compute_interpretation_rewards(descriptions: List[str], critiques: List[Dict[str, float]], 
                                 activation_patterns: List[torch.Tensor]) -> List[float]:
    """
    Compute rewards for feature interpretation quality based on constitutional principles.
    
    Args:
        descriptions: List of generated interpretations
        critiques: List of critique scores from the critique model
        activation_patterns: List of activation patterns being interpreted
        
    Returns:
        List of reward values for each interpretation
    """
    
    rewards = []
    for description, critique, activation in zip(descriptions, critiques, activation_patterns):
        # Extract individual scores
        accuracy_score = critique.get("accuracy", 0.0)
        relevance_score = critique.get("relevance", 0.0)
        clarity_score = critique.get("clarity", 0.0)
        completeness_score = critique.get("completeness", 0.0)
        predictive_score = critique.get("predictive_power", 0.0)
        
        # Weighted combination of scores
        # These weights can be tuned based on importance
        reward = (
            0.3 * accuracy_score +      # How accurately it captures features
            0.25 * relevance_score +    # How relevant the interpretation is
            0.2 * clarity_score +       # How clear and understandable
            0.15 * completeness_score + # How comprehensive the interpretation
            0.1 * predictive_score      # How well it predicts behavior
        )
        
        rewards.append(reward)
    
    return rewards

def compute_persona_rewards(control_strategies: List[str], outputs: List[str], 
                           critiques: List[Dict[str, float]], target_personas: List[str]) -> List[float]:
    """
    Compute rewards for persona impersonation quality based on constitutional principles.
    
    Args:
        control_strategies: List of control strategies applied
        outputs: List of model outputs after control application
        critiques: List of critique scores from the critique model
        target_personas: List of target personas to impersonate
        
    Returns:
        List of reward values for each control strategy
    """
    
    rewards = []
    for strategy, output, critique, persona in zip(control_strategies, outputs, critiques, target_personas):
        # Extract individual scores
        accuracy_score = critique.get("persona_accuracy", 0.0)
        consistency_score = critique.get("consistency", 0.0)
        authenticity_score = critique.get("authenticity", 0.0)
        naturalness_score = critique.get("naturalness", 0.0)
        adaptability_score = critique.get("adaptability", 0.0)
        coherence_score = critique.get("coherence", 0.0)
        
        # Weighted combination of scores
        # These weights can be tuned based on importance
        reward = (
            0.25 * accuracy_score +     # How accurately it impersonates the persona
            0.2 * consistency_score +   # How consistent the persona behavior is
            0.2 * authenticity_score +  # How authentic the persona appears
            0.15 * naturalness_score +  # How natural the persona expression is
            0.1 * adaptability_score +  # How well it adapts to context
            0.1 * coherence_score       # How coherent the persona is throughout
        )
        
        rewards.append(reward)
    
    return rewards

def compute_constitutional_loss(base_loss: torch.Tensor, constitutional_rewards: List[float], 
                              alpha: float = 0.5) -> torch.Tensor:
    """
    Combine base loss with constitutional rewards.
    
    Args:
        base_loss: Original training loss
        constitutional_rewards: List of constitutional reward values
        alpha: Weight for constitutional component (0-1)
        
    Returns:
        Combined loss tensor
    """
    
    # Convert rewards to tensor
    reward_tensor = torch.tensor(constitutional_rewards, dtype=torch.float32, device=base_loss.device)
    
    # Normalize rewards to 0-1 range
    if reward_tensor.max() > reward_tensor.min():
        reward_tensor = (reward_tensor - reward_tensor.min()) / (reward_tensor.max() - reward_tensor.min())
    
    # Convert to loss (higher reward = lower loss)
    constitutional_loss = 1.0 - reward_tensor.mean()
    
    # Combine with base loss
    combined_loss = (1 - alpha) * base_loss + alpha * constitutional_loss
    
    return combined_loss

def compute_reading_rewards(descriptions: List[str], critiques: List[Dict[str, float]]) -> List[float]:
    """
    Compute rewards for reading operations (feature interpretation).
    
    Args:
        descriptions: List of generated descriptions
        critiques: List of critique scores
        
    Returns:
        List of reward values
    """
    
    rewards = []
    for description, critique in zip(descriptions, critiques):
        # Extract scores
        accuracy_score = critique.get("accuracy", 0.0)
        relevance_score = critique.get("relevance", 0.0)
        clarity_score = critique.get("clarity", 0.0)
        completeness_score = critique.get("completeness", 0.0)
        predictive_score = critique.get("predictive_power", 0.0)
        
        # Compute reward
        reward = (
            0.3 * accuracy_score +
            0.25 * relevance_score +
            0.2 * clarity_score +
            0.15 * completeness_score +
            0.1 * predictive_score
        )
        
        rewards.append(reward)
    
    return rewards

def compute_control_rewards(control_strategies: List[str], outputs: List[str], 
                           critiques: List[Dict[str, float]]) -> List[float]:
    """
    Compute rewards for control operations (persona impersonation).
    
    Args:
        control_strategies: List of control strategies
        outputs: List of controlled outputs
        critiques: List of critique scores
        
    Returns:
        List of reward values
    """
    
    rewards = []
    for strategy, output, critique in zip(control_strategies, outputs, critiques):
        # Extract scores
        accuracy_score = critique.get("persona_accuracy", 0.0)
        consistency_score = critique.get("consistency", 0.0)
        authenticity_score = critique.get("authenticity", 0.0)
        naturalness_score = critique.get("naturalness", 0.0)
        adaptability_score = critique.get("adaptability", 0.0)
        coherence_score = critique.get("coherence", 0.0)
        
        # Compute reward
        reward = (
            0.25 * accuracy_score +
            0.2 * consistency_score +
            0.2 * authenticity_score +
            0.15 * naturalness_score +
            0.1 * adaptability_score +
            0.1 * coherence_score
        )
        
        rewards.append(reward)
    
    return rewards

def normalize_rewards(rewards: List[float]) -> List[float]:
    """
    Normalize rewards to 0-1 range.
    
    Args:
        rewards: List of reward values
        
    Returns:
        List of normalized reward values
    """
    
    if not rewards:
        return rewards
    
    rewards_array = np.array(rewards)
    min_reward = rewards_array.min()
    max_reward = rewards_array.max()
    
    if max_reward > min_reward:
        normalized_rewards = (rewards_array - min_reward) / (max_reward - min_reward)
    else:
        normalized_rewards = np.ones_like(rewards_array)
    
    return normalized_rewards.tolist()

def compute_combined_rewards(reading_rewards: List[float], control_rewards: List[float], 
                           reading_weight: float = 0.5) -> List[float]:
    """
    Compute combined rewards for operations that involve both reading and control.
    
    Args:
        reading_rewards: Rewards for reading operations
        control_rewards: Rewards for control operations
        reading_weight: Weight for reading component (0-1)
        
    Returns:
        List of combined reward values
    """
    
    if len(reading_rewards) != len(control_rewards):
        raise ValueError("Reading and control rewards must have the same length")
    
    combined_rewards = []
    for reading_reward, control_reward in zip(reading_rewards, control_rewards):
        combined_reward = reading_weight * reading_reward + (1 - reading_weight) * control_reward
        combined_rewards.append(combined_reward)
    
    return combined_rewards

def compute_constitutional_alignment_score(descriptions: List[str], critiques: List[Dict[str, float]], 
                                        constitution: List[str]) -> float:
    """
    Compute overall constitutional alignment score.
    
    Args:
        descriptions: List of generated descriptions
        critiques: List of critique scores
        constitution: List of constitutional principles
        
    Returns:
        Overall alignment score (0-1)
    """
    
    if not descriptions or not critiques:
        return 0.0
    
    # Compute rewards
    rewards = compute_reading_rewards(descriptions, critiques)
    
    # Return average reward as alignment score
    return np.mean(rewards)

def compute_persona_alignment_score(control_strategies: List[str], outputs: List[str], 
                                  critiques: List[Dict[str, float]], 
                                  constitution: List[str]) -> float:
    """
    Compute overall persona alignment score.
    
    Args:
        control_strategies: List of control strategies
        outputs: List of controlled outputs
        critiques: List of critique scores
        constitution: List of constitutional principles
        
    Returns:
        Overall alignment score (0-1)
    """
    
    if not control_strategies or not critiques:
        return 0.0
    
    # Compute rewards
    rewards = compute_control_rewards(control_strategies, outputs, critiques)
    
    # Return average reward as alignment score
    return np.mean(rewards) 