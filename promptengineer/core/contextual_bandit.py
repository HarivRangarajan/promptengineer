"""
Contextual bandit system for optimizing technique selection.

Uses contextual bandit algorithms to learn which prompt engineering techniques
work best for different types of tasks and contexts.
"""

import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import pickle

from ..techniques.base import PromptContext
from ..techniques.registry import TechniqueRegistry


@dataclass
class ContextualFeatures:
    """Features extracted from prompt context for bandit algorithm."""
    domain: str = "unknown"
    task_complexity: float = 0.5  # 0-1 scale
    has_examples: bool = False
    num_constraints: int = 0
    target_audience_type: str = "general"
    task_type: str = "unknown"  # classification, generation, reasoning, etc.
    
    def to_vector(self) -> np.ndarray:
        """Convert features to numerical vector for ML algorithms."""
        # Simple feature encoding - could be more sophisticated
        domain_encoding = hash(self.domain) % 100 / 100.0
        audience_encoding = hash(self.target_audience_type) % 100 / 100.0
        task_encoding = hash(self.task_type) % 100 / 100.0
        
        return np.array([
            domain_encoding,
            self.task_complexity,
            float(self.has_examples),
            min(self.num_constraints / 10.0, 1.0),  # Normalize constraints
            audience_encoding,
            task_encoding
        ])


@dataclass 
class BanditAction:
    """Represents an action (technique selection) in the bandit framework."""
    technique_name: str
    context_features: ContextualFeatures
    reward: Optional[float] = None
    timestamp: Optional[str] = None


class ContextualBandit:
    """
    Contextual bandit for learning optimal technique selection.
    
    Uses epsilon-greedy with contextual features to balance exploration
    and exploitation when selecting prompt engineering techniques.
    """
    
    def __init__(self, epsilon: float = 0.1, decay_rate: float = 0.99):
        """
        Initialize contextual bandit.
        
        Args:
            epsilon: Exploration rate (0-1)
            decay_rate: Rate at which epsilon decays over time
        """
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.decay_rate = decay_rate
        self.registry = TechniqueRegistry()
        
        # Track actions and rewards
        self.action_history: List[BanditAction] = []
        self.technique_rewards: Dict[str, List[float]] = defaultdict(list)
        self.context_technique_rewards: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        
        # Simple linear model weights for each technique
        self.technique_weights: Dict[str, np.ndarray] = {}
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for each technique."""
        feature_dim = 6  # Based on ContextualFeatures.to_vector()
        for technique_name in self.registry.list_techniques():
            self.technique_weights[technique_name] = np.random.normal(0, 0.1, feature_dim)
    
    def extract_features(self, context: PromptContext) -> ContextualFeatures:
        """Extract features from PromptContext for bandit algorithm."""
        # Estimate task complexity based on description length and keywords
        complexity_indicators = ['complex', 'difficult', 'multi-step', 'reasoning', 'analysis']
        task_complexity = min(len(context.task_description) / 500.0, 1.0)  # Length-based
        for indicator in complexity_indicators:
            if indicator in context.task_description.lower():
                task_complexity = min(task_complexity + 0.2, 1.0)
        
        # Determine task type from description
        task_type = "generation"  # default
        if any(word in context.task_description.lower() for word in ['classify', 'categorize', 'label']):
            task_type = "classification"
        elif any(word in context.task_description.lower() for word in ['reason', 'think', 'analyze', 'solve']):
            task_type = "reasoning"
        elif any(word in context.task_description.lower() for word in ['summarize', 'extract', 'identify']):
            task_type = "extraction"
        
        return ContextualFeatures(
            domain=context.domain or "unknown",
            task_complexity=task_complexity,
            has_examples=bool(context.examples),
            num_constraints=len(context.constraints) if context.constraints else 0,
            target_audience_type=context.target_audience or "general",
            task_type=task_type
        )
    
    def select_technique(self, context: PromptContext, available_techniques: Optional[List[str]] = None) -> str:
        """
        Select best technique using contextual bandit algorithm.
        
        Args:
            context: Prompt context
            available_techniques: Optional list to restrict technique choices
            
        Returns:
            Selected technique name
        """
        if available_techniques is None:
            available_techniques = self.registry.list_techniques()
        
        if not available_techniques:
            raise ValueError("No techniques available")
        
        features = self.extract_features(context)
        feature_vector = features.to_vector()
        
        # Epsilon-greedy selection with contextual rewards
        if np.random.random() < self.epsilon:
            # Explore: random selection
            selected_technique = np.random.choice(available_techniques)
        else:
            # Exploit: select technique with highest predicted reward
            best_technique = None
            best_score = -float('inf')
            
            for technique in available_techniques:
                if technique in self.technique_weights:
                    predicted_reward = np.dot(self.technique_weights[technique], feature_vector)
                else:
                    # New technique, initialize weights
                    self.technique_weights[technique] = np.random.normal(0, 0.1, len(feature_vector))
                    predicted_reward = 0
                
                if predicted_reward > best_score:
                    best_score = predicted_reward
                    best_technique = technique
            
            selected_technique = best_technique or available_techniques[0]
        
        # Record the action (without reward yet)
        action = BanditAction(
            technique_name=selected_technique,
            context_features=features
        )
        self.action_history.append(action)
        
        # Decay epsilon
        self.epsilon = max(0.01, self.epsilon * self.decay_rate)
        
        return selected_technique
    
    def update_reward(self, reward: float, feedback: Optional[str] = None):
        """
        Update the reward for the most recent action.
        
        Args:
            reward: Reward value (typically 0-1 or 1-5 scale)
            feedback: Optional textual feedback
        """
        if not self.action_history:
            raise ValueError("No action to update")
        
        # Update the most recent action
        last_action = self.action_history[-1]
        last_action.reward = reward
        
        # Update technique performance tracking
        technique = last_action.technique_name
        self.technique_rewards[technique].append(reward)
        
        # Update contextual tracking
        context_key = self._context_to_key(last_action.context_features)
        self.context_technique_rewards[context_key][technique].append(reward)
        
        # Update technique performance in registry
        self.registry.update_technique_performance(technique, reward, feedback)
        
        # Update linear model weights using simple gradient update
        self._update_weights(last_action, reward)
    
    def _context_to_key(self, features: ContextualFeatures) -> str:
        """Convert context features to a string key for grouping."""
        return f"{features.domain}_{features.task_type}_{int(features.task_complexity*10)}"
    
    def _update_weights(self, action: BanditAction, reward: float, learning_rate: float = 0.1):
        """Update linear model weights using gradient descent."""
        technique = action.technique_name
        feature_vector = action.context_features.to_vector()
        
        if technique in self.technique_weights:
            # Current prediction
            predicted_reward = np.dot(self.technique_weights[technique], feature_vector)
            
            # Update weights: w = w + lr * (reward - prediction) * features
            error = reward - predicted_reward
            self.technique_weights[technique] += learning_rate * error * feature_vector
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary including contextual analysis."""
        summary = {
            "total_actions": len(self.action_history),
            "current_epsilon": self.epsilon,
            "technique_performance": {},
            "contextual_performance": {},
            "best_techniques_by_context": {}
        }
        
        # Overall technique performance
        for technique, rewards in self.technique_rewards.items():
            if rewards:
                summary["technique_performance"][technique] = {
                    "average_reward": np.mean(rewards),
                    "total_uses": len(rewards),
                    "std_dev": np.std(rewards)
                }
        
        # Contextual performance
        for context_key, technique_rewards in self.context_technique_rewards.items():
            context_summary = {}
            best_technique = None
            best_avg = -1
            
            for technique, rewards in technique_rewards.items():
                if rewards:
                    avg_reward = np.mean(rewards)
                    context_summary[technique] = {
                        "average_reward": avg_reward,
                        "uses": len(rewards)
                    }
                    if avg_reward > best_avg:
                        best_avg = avg_reward
                        best_technique = technique
            
            summary["contextual_performance"][context_key] = context_summary
            if best_technique:
                summary["best_techniques_by_context"][context_key] = best_technique
        
        return summary
    
    def save_state(self, filepath: str):
        """Save bandit state to file."""
        state = {
            "epsilon": self.epsilon,
            "initial_epsilon": self.initial_epsilon,
            "decay_rate": self.decay_rate,
            "action_history": [asdict(action) for action in self.action_history],
            "technique_rewards": dict(self.technique_rewards),
            "context_technique_rewards": dict(self.context_technique_rewards),
            "technique_weights": {k: v.tolist() for k, v in self.technique_weights.items()}
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load_state(self, filepath: str):
        """Load bandit state from file."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.epsilon = state["epsilon"]
        self.initial_epsilon = state["initial_epsilon"] 
        self.decay_rate = state["decay_rate"]
        
        # Reconstruct action history
        self.action_history = [
            BanditAction(
                technique_name=action["technique_name"],
                context_features=ContextualFeatures(**action["context_features"]),
                reward=action.get("reward"),
                timestamp=action.get("timestamp")
            )
            for action in state["action_history"]
        ]
        
        self.technique_rewards = defaultdict(list, state["technique_rewards"])
        self.context_technique_rewards = defaultdict(lambda: defaultdict(list), state["context_technique_rewards"])
        self.technique_weights = {k: np.array(v) for k, v in state["technique_weights"].items()} 