"""
Chain of Thought prompting technique implementation.

Based on the paper "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
by Wei et al. (2022).
"""

from .base import BaseTechnique, PromptContext


class ChainOfThoughtTechnique(BaseTechnique):
    """
    Chain of Thought prompting technique.
    
    Encourages step-by-step reasoning by asking the model to show its work
    and think through problems systematically.
    """
    
    def __init__(self):
        super().__init__(
            name="chain_of_thought",
            description="Prompts the model to show step-by-step reasoning and thinking process"
        )
    
    def get_meta_prompt(self, context: PromptContext) -> str:
        """Generate meta prompt for creating Chain of Thought prompts."""
        
        context_info = self.format_context(context)
        
        meta_prompt = f"""You are an expert prompt engineer specializing in Chain of Thought (CoT) prompting. Your task is to create an optimized prompt that encourages step-by-step reasoning and explicit thinking processes.

CHAIN OF THOUGHT TECHNIQUE PRINCIPLES:
1. Encourage explicit step-by-step reasoning
2. Ask the model to "think out loud" or "show your work"
3. Break complex problems into smaller, manageable steps
4. Use phrases like "Let's think step by step", "First..., then..., finally..."
5. Include examples that demonstrate the reasoning process when possible
6. Make intermediate steps visible and traceable

CONTEXT INFORMATION:
{context_info}

INSTRUCTIONS:
Create a prompt that will effectively apply Chain of Thought reasoning to the given task. The prompt should:

1. Clearly explain the task and its requirements
2. Explicitly instruct the model to show its reasoning process
3. Suggest a logical sequence of steps or thinking approach
4. Include appropriate CoT trigger phrases
5. If examples are provided in context, incorporate them to demonstrate the reasoning pattern
6. Ensure the prompt is clear, structured, and encourages thorough thinking

IMPORTANT: 
- Focus on the reasoning process, not just the final answer
- Use language that naturally guides step-by-step thinking
- Make sure the prompt is specific to the domain and task provided
- Include clear indicators for when reasoning steps should be shown

Generate the Chain of Thought prompt below:"""

        return meta_prompt
    
    def get_judge_meta_prompt(self, context: PromptContext) -> str:
        """Generate meta prompt for creating Chain of Thought evaluation criteria."""
        
        context_info = self.format_context(context)
        
        judge_meta_prompt = f"""You are an expert in evaluating Chain of Thought (CoT) reasoning quality. Your task is to create evaluation criteria and a judge prompt that can assess how well a response demonstrates step-by-step reasoning.

CONTEXT INFORMATION:
{context_info}

CHAIN OF THOUGHT EVALUATION PRINCIPLES:
1. Reasoning Clarity: Are the thinking steps clear and logical?
2. Step Completeness: Are all necessary reasoning steps present?
3. Logical Flow: Do the steps follow logically from one to another?
4. Correctness: Are the reasoning steps factually accurate?
5. Relevance: Do the reasoning steps address the actual question/task?
6. Depth: Is the reasoning sufficiently detailed for the complexity of the task?

INSTRUCTIONS:
Create a comprehensive evaluation prompt for judging Chain of Thought responses. Include:

1. Clear evaluation criteria specific to CoT reasoning
2. Scoring rubric (e.g., 1-5 scale or pass/fail)
3. Examples of good vs. poor reasoning patterns
4. Specific things to look for in step-by-step reasoning
5. How to handle cases where reasoning is partially correct
6. Guidelines for assessing reasoning quality vs. final answer correctness

The judge prompt should be able to:
- Identify presence/absence of step-by-step reasoning
- Evaluate logical consistency across reasoning steps
- Assess whether reasoning leads to appropriate conclusions
- Detect common reasoning errors or gaps
- Account for domain-specific reasoning requirements

FORMAT: Provide the judge prompt that can evaluate CoT responses with clear, actionable criteria.

Generate the Chain of Thought evaluation prompt below:"""

        return judge_meta_prompt 