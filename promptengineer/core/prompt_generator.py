"""
Main prompt generation engine.
"""

import json
from typing import Optional, Dict, Any, List
from openai import OpenAI

from ..techniques.base import BaseTechnique, PromptContext, GeneratedPrompt
from ..techniques.registry import TechniqueRegistry


class PromptGenerator:
    """
    Main prompt generation engine that uses various techniques to create optimized prompts.
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4o", temperature: float = 0.7):
        """
        Initialize the PromptGenerator.
        
        Args:
            api_key: OpenAI API key
            model: Model to use for meta-prompt processing
            temperature: Temperature for LLM calls
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.registry = TechniqueRegistry()
    
    def generate_prompt(
        self, 
        context: PromptContext,
        technique: Optional[str] = None,
        auto_select: bool = False
    ) -> GeneratedPrompt:
        """
        Generate an optimized prompt using the specified technique.
        
        Args:
            context: Context information for prompt generation
            technique: Specific technique to use (e.g., "chain_of_thought")
            auto_select: If True, automatically select best technique
            
        Returns:
            Generated prompt with metadata
        """
        # Select technique
        if auto_select:
            selected_technique = self.registry.get_best_technique()
        elif technique:
            selected_technique = self.registry.get_technique(technique)
            if not selected_technique:
                raise ValueError(f"Unknown technique: {technique}")
        else:
            # Default to chain of thought
            selected_technique = self.registry.get_technique("chain_of_thought")
        
        # Generate meta prompt
        meta_prompt = selected_technique.get_meta_prompt(context)
        
        # Call LLM to generate actual prompt
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert prompt engineer."},
                {"role": "user", "content": meta_prompt}
            ],
            temperature=self.temperature
        )
        
        generated_prompt_text = response.choices[0].message.content
        
        # Create GeneratedPrompt object
        generated_prompt = GeneratedPrompt(
            prompt=generated_prompt_text,
            technique=selected_technique.name,
            meta_prompt_used=meta_prompt,
            generation_parameters={
                "model": self.model,
                "temperature": self.temperature,
                "auto_select": auto_select
            }
        )
        
        return generated_prompt
    
    def list_available_techniques(self) -> List[str]:
        """List all available techniques."""
        return self.registry.list_techniques()
    
    def get_technique_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all available techniques."""
        return self.registry.get_technique_descriptions()
    
    def register_custom_technique(self, technique: BaseTechnique):
        """Register a custom technique."""
        self.registry.register_technique(technique)
    
    def update_technique_performance(self, technique_name: str, score: float, feedback: Optional[str] = None):
        """Update performance metrics for a technique."""
        self.registry.update_technique_performance(technique_name, score, feedback)
    
    def get_performance_summary(self) -> Dict[str, Dict]:
        """Get performance summary for all techniques."""
        return self.registry.get_performance_summary()
    
    def create_context(
        self,
        task_description: str,
        domain: Optional[str] = None,
        examples: Optional[List[Dict[str, Any]]] = None,
        constraints: Optional[List[str]] = None,
        target_audience: Optional[str] = None,
        success_criteria: Optional[List[str]] = None,
        **kwargs
    ) -> PromptContext:
        """
        Convenience method to create a PromptContext.
        
        Args:
            task_description: Description of the task
            domain: Domain/field (e.g., "medical", "legal", "education")
            examples: Example inputs/outputs for the task
            constraints: List of constraints or requirements
            target_audience: Target audience for the prompts
            success_criteria: Success criteria for evaluation
            **kwargs: Additional context information
            
        Returns:
            PromptContext object
        """
        return PromptContext(
            task_description=task_description,
            domain=domain,
            examples=examples,
            constraints=constraints,
            target_audience=target_audience,
            success_criteria=success_criteria,
            additional_context=kwargs
        ) 