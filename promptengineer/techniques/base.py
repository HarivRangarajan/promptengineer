"""
Base classes for prompt engineering techniques.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class PromptContext:
    """Context information for prompt generation."""
    task_description: str
    domain: Optional[str] = None
    examples: Optional[List[Dict[str, Any]]] = None
    constraints: Optional[List[str]] = None
    target_audience: Optional[str] = None
    success_criteria: Optional[List[str]] = None
    additional_context: Optional[Dict[str, Any]] = None


@dataclass
class GeneratedPrompt:
    """Generated prompt with metadata."""
    prompt: str
    technique: str
    meta_prompt_used: str
    generation_parameters: Dict[str, Any]
    confidence_score: Optional[float] = None


@dataclass
class JudgePrompt:
    """Generated judge prompt with evaluation criteria."""
    prompt: str
    criteria: List[str]
    scoring_method: str
    technique_evaluated: str
    meta_prompt_used: str


class BaseTechnique(ABC):
    """Base class for all prompt engineering techniques."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.performance_history = []
    
    @abstractmethod
    def get_meta_prompt(self, context: PromptContext) -> str:
        """Generate meta prompt for this technique given the context."""
        pass
    
    @abstractmethod
    def get_judge_meta_prompt(self, context: PromptContext) -> str:
        """Generate meta prompt for creating judge evaluation criteria."""
        pass
    
    def format_context(self, context: PromptContext) -> str:
        """Format context information for meta prompt inclusion."""
        formatted = f"Task: {context.task_description}\n"
        
        if context.domain:
            formatted += f"Domain: {context.domain}\n"
        
        if context.target_audience:
            formatted += f"Target Audience: {context.target_audience}\n"
        
        if context.constraints:
            formatted += f"Constraints: {', '.join(context.constraints)}\n"
        
        if context.success_criteria:
            formatted += f"Success Criteria: {', '.join(context.success_criteria)}\n"
        
        if context.examples:
            formatted += f"Examples: {len(context.examples)} provided\n"
            for i, example in enumerate(context.examples[:3]):  # Show first 3
                formatted += f"  Example {i+1}: {str(example)[:100]}...\n"
        
        return formatted
    
    def update_performance(self, score: float, feedback: Optional[str] = None):
        """Update performance history for this technique."""
        self.performance_history.append({
            'score': score,
            'feedback': feedback,
            'timestamp': None  # Could add timestamp
        })
    
    def get_average_performance(self) -> float:
        """Get average performance score for this technique."""
        if not self.performance_history:
            return 0.0
        return sum(h['score'] for h in self.performance_history) / len(self.performance_history) 