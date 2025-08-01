"""
Registry for managing prompt engineering techniques.
"""

from typing import Dict, List, Type, Optional
from .base import BaseTechnique
from .chain_of_thought import ChainOfThoughtTechnique


class TechniqueRegistry:
    """Registry for managing and discovering prompt engineering techniques."""
    
    def __init__(self):
        self._techniques: Dict[str, BaseTechnique] = {}
        self._register_default_techniques()
    
    def _register_default_techniques(self):
        """Register all default techniques."""
        # Import here to avoid circular imports
        from .chain_of_thought import ChainOfThoughtTechnique
        from .chain_of_thoughtlessness import ChainOfThoughtlessnessTechnique
        from .chain_of_draft import ChainOfDraftTechnique
        
        # Register Chain of Thought
        cot = ChainOfThoughtTechnique()
        self._techniques[cot.name] = cot
        
        # Register placeholder techniques (will be implemented based on papers)
        self._techniques["chain_of_thoughtlessness"] = ChainOfThoughtlessnessTechnique()
        self._techniques["chain_of_draft"] = ChainOfDraftTechnique()
    
    def register_technique(self, technique: BaseTechnique):
        """Register a new technique."""
        self._techniques[technique.name] = technique
    
    def get_technique(self, name: str) -> Optional[BaseTechnique]:
        """Get a technique by name."""
        return self._techniques.get(name)
    
    def list_techniques(self) -> List[str]:
        """List all available technique names."""
        return list(self._techniques.keys())
    
    def get_technique_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all techniques."""
        return {name: tech.description for name, tech in self._techniques.items()}
    
    def get_best_technique(self, context_type: Optional[str] = None) -> BaseTechnique:
        """
        Get the best performing technique, optionally filtered by context type.
        
        For now, returns the technique with highest average performance.
        In the future, this could use more sophisticated selection logic.
        """
        if not self._techniques:
            raise ValueError("No techniques registered")
        
        # Find technique with best average performance
        best_technique = None
        best_score = -1
        
        for technique in self._techniques.values():
            score = technique.get_average_performance()
            if score > best_score:
                best_score = score
                best_technique = technique
        
        # If no technique has performance history, return Chain of Thought as default
        if best_technique is None:
            return self._techniques.get("chain_of_thought", list(self._techniques.values())[0])
        
        return best_technique
    
    def update_technique_performance(self, technique_name: str, score: float, feedback: Optional[str] = None):
        """Update performance for a specific technique."""
        if technique_name in self._techniques:
            self._techniques[technique_name].update_performance(score, feedback)
    
    def get_performance_summary(self) -> Dict[str, Dict]:
        """Get performance summary for all techniques."""
        summary = {}
        for name, technique in self._techniques.items():
            summary[name] = {
                'average_score': technique.get_average_performance(),
                'total_evaluations': len(technique.performance_history),
                'description': technique.description
            }
        return summary 