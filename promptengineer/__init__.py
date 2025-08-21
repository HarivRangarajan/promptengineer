"""
PromptEngineer: A modular library for automated prompt engineering and LLM-as-a-Judge evaluation.

This library helps users create optimized prompts for various tasks using research-backed techniques
and provides automated evaluation capabilities through LLM-as-a-Judge systems.
"""

__version__ = "0.1.0"
__author__ = "PromptEngineer Team"

from .core.prompt_generator import PromptGenerator
from .core.judge_generator import JudgeGenerator
from .core.contextual_bandit import ContextualBandit
from .core.pipeline import PromptPipeline
from .core.demonstration_generator import DemonstrationGenerator
from .core.feedback_integrator import FeedbackIntegrator
from .techniques.registry import TechniqueRegistry

# Main API
__all__ = [
    "PromptGenerator",
    "JudgeGenerator", 
    "ContextualBandit",
    "PromptPipeline",
    "DemonstrationGenerator",
    "FeedbackIntegrator",
    "TechniqueRegistry"
]

# Convenience function for quick setup
def create_prompt_engineer(api_key: str, model: str = "gpt-4o"):
    """Create a PromptGenerator with default settings."""
    return PromptGenerator(api_key=api_key, model=model)

def create_judge_generator(api_key: str, model: str = "gpt-4o"):
    """Create a JudgeGenerator with default settings."""
    return JudgeGenerator(api_key=api_key, model=model)

def create_demonstration_generator(api_key: str, model: str = "gpt-4o", output_dir: str = "demonstration_output"):
    """Create a DemonstrationGenerator with default settings."""
    return DemonstrationGenerator(api_key=api_key, model=model, output_dir=output_dir) 