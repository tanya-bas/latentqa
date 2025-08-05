"""
Critique models for Constitutional AI in LatentQA.
Evaluates quality of feature interpretations and persona impersonations using AI.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import numpy as np
from typing import List, Dict, Tuple, Any

class FeatureInterpretationCritique(nn.Module):
    """Critique model for evaluating feature interpretation quality."""
    
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Freeze the base model
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Add critique head
        self.critique_head = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 5)  # 5 evaluation criteria
        )
        
    def forward(self, interpretation: str, activation_pattern: torch.Tensor, principle: str) -> Dict[str, float]:
        """Evaluate interpretation quality based on constitutional principle."""
        
        # Encode interpretation and principle
        inputs = self.tokenizer(
            f"Principle: {principle}\nInterpretation: {interpretation}",
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
            
        # Get critique scores
        critique_scores = self.critique_head(hidden_states)
        scores = torch.sigmoid(critique_scores)
        
        return {
            "accuracy": scores[0, 0].item(),
            "relevance": scores[0, 1].item(),
            "clarity": scores[0, 2].item(),
            "completeness": scores[0, 3].item(),
            "predictive_power": scores[0, 4].item()
        }
    
    def evaluate(self, interpretations: List[str], activation_patterns: List[torch.Tensor], 
                 constitution: List[str]) -> List[Dict[str, float]]:
        """Evaluate multiple interpretations against constitutional principles."""
        
        critiques = []
        for interpretation, activation_pattern in zip(interpretations, activation_patterns):
            principle_scores = []
            for principle in constitution:
                scores = self.forward(interpretation, activation_pattern, principle)
                principle_scores.append(scores)
            
            # Aggregate scores across principles
            avg_scores = {}
            for key in principle_scores[0].keys():
                avg_scores[key] = np.mean([ps[key] for ps in principle_scores])
            
            critiques.append(avg_scores)
        
        return critiques

class PersonaImpersonationCritique(nn.Module):
    """Critique model for evaluating persona impersonation quality."""
    
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Freeze the base model
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Add critique head
        self.critique_head = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 6)  # 6 evaluation criteria
        )
        
    def forward(self, control_strategy: str, output: str, target_persona: str, principle: str) -> Dict[str, float]:
        """Evaluate persona impersonation quality based on constitutional principle."""
        
        # Encode strategy, output, persona, and principle
        inputs = self.tokenizer(
            f"Principle: {principle}\nTarget Persona: {target_persona}\nControl Strategy: {control_strategy}\nOutput: {output}",
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
            
        # Get critique scores
        critique_scores = self.critique_head(hidden_states)
        scores = torch.sigmoid(critique_scores)
        
        return {
            "persona_accuracy": scores[0, 0].item(),
            "consistency": scores[0, 1].item(),
            "authenticity": scores[0, 2].item(),
            "naturalness": scores[0, 3].item(),
            "adaptability": scores[0, 4].item(),
            "coherence": scores[0, 5].item()
        }
    
    def evaluate(self, control_strategies: List[str], outputs: List[str], 
                 target_personas: List[str], constitution: List[str]) -> List[Dict[str, float]]:
        """Evaluate multiple control strategies against constitutional principles."""
        
        critiques = []
        for strategy, output, persona in zip(control_strategies, outputs, target_personas):
            principle_scores = []
            for principle in constitution:
                scores = self.forward(strategy, output, persona, principle)
                principle_scores.append(scores)
            
            # Aggregate scores across principles
            avg_scores = {}
            for key in principle_scores[0].keys():
                avg_scores[key] = np.mean([ps[key] for ps in principle_scores])
            
            critiques.append(avg_scores)
        
        return critiques

class AICritiqueEvaluator:
    """AI-based critique evaluator using a language model."""
    
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
    def evaluate_interpretation(self, interpretation: str, principle: str) -> Dict[str, float]:
        """Use AI to evaluate interpretation against a principle."""
        
        # Create evaluation prompt
        prompt = f"""Evaluate this interpretation against the principle:

Principle: {principle}
Interpretation: {interpretation}

Rate the following aspects (0-1):
- Accuracy: How accurately does it capture the model's features?
- Relevance: How relevant is the interpretation to the principle?
- Clarity: How clear and understandable is the interpretation?
- Completeness: How comprehensive is the analysis?
- Predictive Power: How well does it predict model behavior?

Provide scores as: accuracy=X,relevance=Y,clarity=Z,completeness=W,predictive_power=V"""

        # Use model to generate evaluation
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Extract evaluation from model output
            # This is a simplified approach - in practice you'd need more sophisticated parsing
            evaluation_text = self.tokenizer.decode(outputs.last_hidden_state[0], skip_special_tokens=True)
        
        # Parse scores (simplified parsing)
        scores = self._parse_evaluation_scores(evaluation_text)
        
        return scores
    
    def evaluate_persona(self, strategy: str, output: str, persona: str, principle: str) -> Dict[str, float]:
        """Use AI to evaluate persona impersonation against a principle."""
        
        # Create evaluation prompt
        prompt = f"""Evaluate this persona impersonation against the principle:

Principle: {principle}
Target Persona: {persona}
Control Strategy: {strategy}
Output: {output}

Rate the following aspects (0-1):
- Persona Accuracy: How well does the output match the target persona?
- Consistency: How consistent is the persona behavior?
- Authenticity: How authentic does the persona appear?
- Naturalness: How natural is the persona expression?
- Adaptability: How well does it adapt to context?
- Coherence: How coherent is the persona throughout?

Provide scores as: persona_accuracy=X,consistency=Y,authenticity=Z,naturalness=W,adaptability=V,coherence=U"""

        # Use model to generate evaluation
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            evaluation_text = self.tokenizer.decode(outputs.last_hidden_state[0], skip_special_tokens=True)
        
        # Parse scores
        scores = self._parse_evaluation_scores(evaluation_text)
        
        return scores
    
    def _parse_evaluation_scores(self, evaluation_text: str) -> Dict[str, float]:
        """Parse evaluation scores from model output."""
        
        # Default scores
        default_scores = {
            "accuracy": 0.5, "relevance": 0.5, "clarity": 0.5, "completeness": 0.5, "predictive_power": 0.5,
            "persona_accuracy": 0.5, "consistency": 0.5, "authenticity": 0.5, "naturalness": 0.5, "adaptability": 0.5, "coherence": 0.5
        }
        
        try:
            # Extract scores from text
            scores = {}
            for line in evaluation_text.split('\n'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip().lower()
                    try:
                        scores[key] = float(value.strip())
                    except ValueError:
                        continue
            
            # Update default scores with parsed values
            for key in default_scores:
                if key in scores:
                    default_scores[key] = scores[key]
            
        except Exception:
            pass
        
        return default_scores

def create_feature_critique_model(model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct") -> FeatureInterpretationCritique:
    """Create a feature interpretation critique model."""
    return FeatureInterpretationCritique(model_name)

def create_persona_critique_model(model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct") -> PersonaImpersonationCritique:
    """Create a persona impersonation critique model."""
    return PersonaImpersonationCritique(model_name)

def create_ai_critique_evaluator(model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct") -> AICritiqueEvaluator:
    """Create an AI-based critique evaluator."""
    return AICritiqueEvaluator(model_name) 