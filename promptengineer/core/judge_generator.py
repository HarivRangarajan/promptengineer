"""
LLM-as-a-Judge prompt generation engine.
"""

import json
from typing import Optional, Dict, Any, List
from openai import OpenAI

from ..techniques.base import BaseTechnique, PromptContext, JudgePrompt
from ..techniques.registry import TechniqueRegistry


class JudgeGenerator:
    """
    Generator for LLM-as-a-Judge evaluation prompts.
    
    Creates evaluation prompts that can assess the quality of responses
    generated using different prompt engineering techniques.
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4o", temperature: float = 0.3):
        """
        Initialize the JudgeGenerator.
        
        Args:
            api_key: OpenAI API key
            model: Model to use for meta-prompt processing
            temperature: Temperature for LLM calls (lower for more consistent evaluation)
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.registry = TechniqueRegistry()
    
    def generate_judge_prompt(
        self,
        context: PromptContext,
        technique_to_evaluate: str,
        scoring_method: str = "1-5_scale"
    ) -> JudgePrompt:
        """
        Generate an LLM-as-a-Judge evaluation prompt for a specific technique.
        
        Args:
            context: Context information for the task being evaluated
            technique_to_evaluate: The technique whose outputs will be evaluated
            scoring_method: Scoring method (e.g., "1-5_scale", "pass_fail", "categorical")
            
        Returns:
            JudgePrompt object with evaluation criteria
        """
        # Get the technique
        technique = self.registry.get_technique(technique_to_evaluate)
        if not technique:
            raise ValueError(f"Unknown technique: {technique_to_evaluate}")
        
        # Generate judge meta prompt
        judge_meta_prompt = technique.get_judge_meta_prompt(context)
        
        # Add scoring method context
        scoring_context = self._get_scoring_context(scoring_method)
        full_meta_prompt = f"{judge_meta_prompt}\n\n{scoring_context}"
        
        # Call LLM to generate judge prompt
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert in creating evaluation criteria and judge prompts for LLM responses."},
                {"role": "user", "content": full_meta_prompt}
            ],
            temperature=self.temperature
        )
        
        generated_judge_prompt = response.choices[0].message.content
        
        # Extract criteria (this is a simplified extraction - could be more sophisticated)
        criteria = self._extract_criteria(generated_judge_prompt, technique_to_evaluate)
        
        # Create JudgePrompt object
        judge_prompt = JudgePrompt(
            prompt=generated_judge_prompt,
            criteria=criteria,
            scoring_method=scoring_method,
            technique_evaluated=technique_to_evaluate,
            meta_prompt_used=full_meta_prompt
        )
        
        return judge_prompt
    
    def _get_scoring_context(self, scoring_method: str) -> str:
        """Get context information for the specified scoring method."""
        scoring_contexts = {
            "1-5_scale": """
SCORING METHOD: Use a 1-5 scale where:
1 = Poor/Inadequate
2 = Below Average  
3 = Average/Acceptable
4 = Good/Above Average
5 = Excellent/Outstanding

Provide both a numerical score and brief justification.""",
            
            "pass_fail": """
SCORING METHOD: Use a Pass/Fail binary evaluation where:
Pass (1) = Response meets the minimum requirements and criteria
Fail (0) = Response does not meet the requirements or has significant issues

Provide the binary score and detailed reasoning.""",
            
            "categorical": """
SCORING METHOD: Use categorical evaluation with specific categories for different types of issues or strengths.
Define clear categories and assign responses to appropriate categories.
Provide category assignment and explanation."""
        }
        
        return scoring_contexts.get(scoring_method, scoring_contexts["1-5_scale"])
    
    def _extract_criteria(self, judge_prompt: str, technique: str) -> List[str]:
        """Extract evaluation criteria from the generated judge prompt."""
        # This is a simplified implementation
        # In practice, you might want more sophisticated parsing
        
        # Look for common patterns that indicate criteria
        criteria = []
        lines = judge_prompt.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for numbered criteria, bullet points, or criteria-like patterns
            if (line.startswith(('1.', '2.', '3.', '4.', '5.')) or 
                line.startswith(('•', '-', '*')) or
                'criteria' in line.lower() or
                'evaluate' in line.lower()):
                if len(line) > 10:  # Avoid very short lines
                    criteria.append(line)
        
        # If we couldn't extract criteria automatically, provide defaults based on technique
        if not criteria:
            criteria = self._get_default_criteria(technique)
        
        return criteria[:10]  # Limit to 10 criteria
    
    def _get_default_criteria(self, technique: str) -> List[str]:
        """Get default criteria for a technique if extraction fails."""
        default_criteria = {
            "chain_of_thought": [
                "Reasoning clarity and logical flow",
                "Completeness of step-by-step thinking", 
                "Correctness of reasoning steps",
                "Relevance to the original question",
                "Depth and thoroughness of analysis"
            ],
            "chain_of_thoughtlessness": [
                "Direct response quality",
                "Accuracy of final answer",
                "Conciseness and clarity",
                "Relevance to the task"
            ],
            "chain_of_draft": [
                "Quality of iterative improvement",
                "Final output quality",
                "Refinement effectiveness",
                "Overall coherence"
            ]
        }
        
        return default_criteria.get(technique, [
            "Response accuracy",
            "Task completion",
            "Clarity and coherence",
            "Appropriateness for context"
        ])
    
    def evaluate_response(
        self,
        judge_prompt: JudgePrompt,
        response_to_evaluate: str,
        original_input: str
    ) -> Dict[str, Any]:
        """
        Use the generated judge prompt to evaluate a response.
        
        Args:
            judge_prompt: The JudgePrompt to use for evaluation
            response_to_evaluate: The response to be evaluated
            original_input: The original input that generated the response
            
        Returns:
            Evaluation results
        """
        evaluation_input = f"""
ORIGINAL INPUT:
{original_input}

RESPONSE TO EVALUATE:
{response_to_evaluate}

Please evaluate this response according to the criteria provided.
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": judge_prompt.prompt},
                {"role": "user", "content": evaluation_input}
            ],
            temperature=self.temperature
        )
        
        evaluation_result = response.choices[0].message.content
        
        return {
            "evaluation": evaluation_result,
            "judge_prompt_used": judge_prompt.prompt,
            "technique_evaluated": judge_prompt.technique_evaluated,
            "scoring_method": judge_prompt.scoring_method,
            "criteria": judge_prompt.criteria
        } 