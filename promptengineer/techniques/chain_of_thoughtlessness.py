"""
Chain of Thoughtlessness prompting technique implementation.

Based on "Chain of Thoughtlessness? An Analysis of CoT in Planning" by Stechly et al. (2024).
This technique addresses the limitations of Chain of Thought by focusing on direct, 
pattern-matching approaches rather than verbose step-by-step reasoning that often fails to generalize.
"""

from .base import BaseTechnique, PromptContext


class ChainOfThoughtlessnessTechnique(BaseTechnique):
    """
    Chain of Thoughtlessness prompting technique.
    
    Based on research showing that Chain of Thought's effectiveness comes from pattern matching
    rather than true algorithmic reasoning, this technique emphasizes:
    - Direct, concise responses without verbose reasoning
    - Pattern-based problem solving
    - Task-specific optimization over general reasoning
    - Honest acknowledgment of model limitations
    """
    
    def __init__(self):
        super().__init__(
            name="chain_of_thoughtlessness",
            description="Direct, pattern-matching approach that avoids verbose reasoning steps, based on research showing CoT limitations"
        )
    
    def get_meta_prompt(self, context: PromptContext) -> str:
        """Generate meta prompt for creating Chain of Thoughtlessness prompts."""
        
        context_info = self.format_context(context)
        
        meta_prompt = f"""You are an expert prompt engineer specializing in the Chain of Thoughtlessness approach, based on research by Stechly et al. (2024) that revealed the limitations of verbose step-by-step reasoning in Chain of Thought prompting.

CHAIN OF THOUGHTLESSNESS TECHNIQUE PRINCIPLES:
1. Emphasize direct, concise responses over verbose step-by-step reasoning
2. Focus on pattern recognition and matching rather than explicit algorithmic thinking
3. Avoid encouraging the model to "show its work" or explain reasoning steps
4. Recognize that effective prompting often relies on specific pattern matching rather than general reasoning
5. Be task-specific and direct rather than trying to teach general algorithms
6. Acknowledge model limitations rather than pretending to induce complex reasoning
7. Use clear, direct instructions without intermediate reasoning requirements
8. Focus on what actually works (pattern matching) rather than what we wish worked (general reasoning)

RESEARCH INSIGHTS TO INCORPORATE:
- CoT performance improvements often come from pattern matching, not algorithmic learning
- Highly specific prompts work better than general ones, but don't generalize well
- Step-by-step reasoning often fails to scale to larger or more complex problems
- Direct approaches can be more honest about model capabilities and limitations
- Verbose reasoning chains may give false confidence without actual understanding

CONTEXT INFORMATION:
{context_info}

INSTRUCTIONS:
Create a prompt that effectively applies the Chain of Thoughtlessness approach to the given task. The prompt should:

1. Give clear, direct instructions for the task
2. Avoid asking the model to show step-by-step reasoning or "think out loud"
3. Focus on pattern recognition relevant to the specific task
4. Use task-specific knowledge rather than trying to teach general algorithms
5. If examples are provided in context, use them to establish clear patterns without verbose explanations
6. Be concise and direct rather than encouraging lengthy reasoning processes
7. Set realistic expectations about what the model can accomplish
8. Focus on immediate pattern matching rather than complex reasoning chains

IMPORTANT:
- Avoid phrases like "let's think step by step" or "show your work"
- Don't encourage intermediate reasoning steps unless absolutely necessary
- Focus on direct pattern application rather than reasoning explanation
- Be task-specific rather than trying to teach general problem-solving
- Acknowledge what the model can realistically accomplish

Generate the Chain of Thoughtlessness prompt below:"""

        return meta_prompt
    
    def get_judge_meta_prompt(self, context: PromptContext) -> str:
        """Generate meta prompt for creating Chain of Thoughtlessness evaluation criteria."""
        
        context_info = self.format_context(context)
        
        judge_meta_prompt = f"""You are an expert in evaluating Chain of Thoughtlessness responses, based on research by Stechly et al. (2024) that revealed the limitations of verbose reasoning approaches.

CONTEXT INFORMATION:
{context_info}

CHAIN OF THOUGHTLESSNESS EVALUATION PRINCIPLES:
1. Direct Response Quality: Is the response direct and to-the-point without unnecessary verbosity?
2. Pattern Application: Does the response show effective pattern matching relevant to the task?
3. Task Completion: Does the response accomplish the specific task without over-explanation?
4. Conciseness: Is the response appropriately concise rather than verbose?
5. Accuracy: Is the final answer correct regardless of reasoning shown?
6. Realism: Does the response reflect realistic model capabilities rather than pretended reasoning?
7. Specificity: Is the response appropriately specific to the task rather than overly general?

EVALUATION FOCUS:
Unlike Chain of Thought evaluation, this should NOT heavily weight:
- Presence of step-by-step reasoning
- Verbose explanations of thinking process
- Complex reasoning chains
- General algorithmic procedures

Instead, focus on:
- Direct effectiveness for the specific task
- Appropriate use of relevant patterns
- Concise communication
- Honest representation of capabilities
- Task-specific competence

INSTRUCTIONS:
Create a comprehensive evaluation prompt for judging Chain of Thoughtlessness responses. Include:

1. Clear evaluation criteria that prioritize directness over verbosity
2. Scoring methodology appropriate for pattern-matching approaches
3. Guidelines for assessing task-specific effectiveness
4. Methods to evaluate conciseness vs. completeness
5. Criteria for realistic capability assessment
6. How to handle cases where direct answers are more valuable than explained reasoning
7. Assessment of pattern application appropriateness

The judge prompt should be able to:
- Recognize effective direct responses vs. unnecessarily verbose ones
- Evaluate task completion without requiring reasoning explanations
- Assess appropriate pattern usage for the specific domain
- Distinguish between helpful conciseness and unhelpful brevity
- Account for the specific task context and requirements

FORMAT: Provide the judge prompt that can evaluate Chain of Thoughtlessness responses with clear, actionable criteria.

Generate the Chain of Thoughtlessness evaluation prompt below:"""

        return judge_meta_prompt 