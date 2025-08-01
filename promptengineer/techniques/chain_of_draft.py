"""
Chain of Draft prompting technique implementation.

This is a placeholder implementation that will be updated based on the paper
provided by the user.
"""

from .base import BaseTechnique, PromptContext

class ChainOfDraftTechnique(BaseTechnique):
    """
    Chain of Draft (CoD) prompting technique.
    
    Based on "Chain of Draft: Thinking Faster by Writing Less" by Xu et al. (2025).
    This technique generates minimalistic yet informative intermediate reasoning outputs,
    inspired by how humans jot down concise drafts when solving problems.
    
    Key principles:
    - Focus on essential information only (5 words max per step)
    - Abstract away irrelevant contextual details
    - Use symbolic/equation formats when possible
    - Eliminate verbose explanations about problem context
    - Significantly reduce token usage while maintaining accuracy
    """
    
    def __init__(self):
        super().__init__(
            name="chain_of_draft",
            description="Generates minimalistic, draft-like reasoning steps that capture only essential information"
        )
    
    def get_meta_prompt(self, context: PromptContext) -> str:
        """Generate meta prompt for creating Chain of Draft prompts."""
        
        context_info = self.format_context(context)
        
        meta_prompt = f"""You are an expert prompt engineer specializing in Chain of Draft (CoD) prompting. Your task is to create an optimized prompt that encourages minimalistic, draft-like reasoning steps inspired by human cognitive processes.

CHAIN OF DRAFT TECHNIQUE PRINCIPLES:
1. Focus on ESSENTIAL information only - what's truly needed to progress
2. Generate minimalistic reasoning steps (ideally ≤5 words each)
3. Use symbolic representations, equations, or shorthand when possible
4. Abstract away irrelevant contextual details and verbose explanations
5. Each step should be like a "draft note" - minimal but informative
6. Prioritize mathematical operations, key transformations, or critical insights
7. Avoid explaining WHY something is being done, just capture WHAT is being done
8. Use concise formats like equations rather than full sentences

CONTEXT INFORMATION:
{context_info}

EXAMPLES OF GOOD CHAIN OF DRAFT STEPS:
- "20 - x = 12; x = 8" (instead of "To find how many Jason gave away, I need to subtract...")
- "Initial: 100, After: 75" (instead of "Jason started with 100 lollipops and now has 75")
- "Pattern: +2, +4, +6" (instead of "I notice the sequence increases by 2, then 4, then 6")
- "Area = πr²; r = 5" (instead of "To find the area of this circle with radius 5...")

INSTRUCTIONS:
Create a prompt that will effectively apply Chain of Draft reasoning to the given task. The prompt should:

1. Clearly explain the task and its requirements
2. Instruct the model to think step by step using minimal drafts
3. Specify the 5-word limit for reasoning steps
4. Encourage symbolic/equation formats when applicable
5. Request that essential operations and insights be captured concisely
6. Ask for the final answer to be clearly marked

Generate a prompt that balances extreme conciseness with logical flow and accuracy."""

        return meta_prompt

    def get_judge_meta_prompt(self, context: PromptContext) -> str:
        """Generate meta prompt for creating Chain of Draft evaluation criteria."""
        
        context_info = self.format_context(context)
        
        judge_meta_prompt = f"""You are an expert evaluator specializing in Chain of Draft (CoD) prompt assessment. Your task is to create evaluation criteria for judging Chain of Draft prompts and responses, focusing on minimalistic, draft-like reasoning quality.

CONTEXT INFORMATION:
{context_info}

CORE CHAIN OF DRAFT PRINCIPLES TO EVALUATE:
1. **Minimalism**: Each reasoning step contains only essential information (ideally ≤5 words)
2. **Abstraction**: Focuses on key operations/insights, removes irrelevant contextual details
3. **Symbolic Efficiency**: Uses equations, shorthand, or symbolic representations when possible
4. **Draft-like Quality**: Steps feel like "notes to self" rather than explanations to others
5. **Information Density**: High ratio of useful information to total words
6. **Operational Focus**: Emphasizes WHAT is being done rather than WHY

EVALUATION FRAMEWORK:
Create comprehensive evaluation criteria that assess:

**Conciseness & Efficiency (25 points)**
- Are reasoning steps genuinely minimal and draft-like?
- Does it avoid verbose explanations and unnecessary elaboration?
- Is the token usage significantly reduced compared to traditional reasoning?

**Essential Information Capture (25 points)**  
- Do the minimal steps still capture all critical information needed?
- Are key mathematical operations, patterns, or insights preserved?
- Can someone follow the logical flow despite the brevity?

**Symbolic & Abstract Representation (25 points)**
- Does it use equations, shorthand, or symbolic formats effectively?
- Are irrelevant contextual details abstracted away?
- Does it focus on core transformations rather than problem narrative?

**Accuracy & Logical Flow (25 points)**
- Does the concise reasoning lead to correct conclusions?
- Is there a clear logical progression despite the minimalism?
- Are the final answers properly identified and correct?

INSTRUCTIONS:
Generate evaluation criteria that will help assess Chain of Draft effectiveness. Include specific examples of what constitutes good vs. poor Chain of Draft reasoning, and provide clear scoring guidelines that reward extreme conciseness while maintaining accuracy."""

        return judge_meta_prompt 