# PromptEngineer

A modular Python library for human centred prompt engineering.

## Overview

PromptEngineer helps you create optimized prompts for various tasks using research-backed techniques and provides automated evaluation capabilities through LLM-as-a-Judge systems. The library includes a contextual bandit system that learns which techniques work best for different types of tasks.

## Features

- **Multiple Prompt Engineering Techniques**: Chain of Thought, Chain of Thoughtlessness, Chain of Draft, and extensible framework for custom techniques
- **LLM-as-a-Judge Generation**: Automatically create evaluation prompts and criteria for assessing prompt quality
- **Contextual Bandit Optimization**: Learn which techniques work best for different task types and contexts
- **Modular Architecture**: Easy to extend with new techniques and evaluation methods
- **Research-Based**: Built on proven prompt engineering research and best practices

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from promptengineer import PromptGenerator, JudgeGenerator
from promptengineer.techniques.base import PromptContext

# Initialize generators
generator = PromptGenerator(api_key="your-openai-api-key")
judge_generator = JudgeGenerator(api_key="your-openai-api-key")

# Create context for your task
context = PromptContext(
    task_description="Guide users through a complex procedure",
    domain="medical",
    target_audience="patients",
    constraints=["Cannot see visual elements", "Must be safe and accurate"]
)

# Generate optimized prompt using Chain of Thought
prompt = generator.generate_prompt(context, technique="chain_of_thought")
print(prompt.prompt)

# Generate evaluation prompt for the technique
judge_prompt = judge_generator.generate_judge_prompt(
    context=context,
    technique_to_evaluate="chain_of_thought"
)
print(judge_prompt.prompt)
```

## Advanced Usage

### Contextual Bandit Optimization

```python
from promptengineer import ContextualBandit

# Initialize bandit for technique selection
bandit = ContextualBandit(epsilon=0.1)

# Let bandit select the best technique
selected_technique = bandit.select_technique(context)

# Generate prompt with selected technique
prompt = generator.generate_prompt(context, technique=selected_technique)

# After evaluation, provide feedback
evaluation_score = 0.85  # From your evaluation process
bandit.update_reward(evaluation_score)
```

### Auto-Selection Mode

```python
# Automatically select the best technique
prompt = generator.generate_prompt(context, auto_select=True)
```

### Custom Techniques

```python
from promptengineer.techniques.base import BaseTechnique

class MyCustomTechnique(BaseTechnique):
    def __init__(self):
        super().__init__(
            name="my_technique",
            description="My custom prompting technique"
        )
    
    def get_meta_prompt(self, context):
        # Return meta prompt for generating actual prompts
        return f"Create a prompt for: {context.task_description}"
    
    def get_judge_meta_prompt(self, context):
        # Return meta prompt for generating evaluation criteria
        return f"Create evaluation criteria for: {context.task_description}"

# Register custom technique
generator.register_custom_technique(MyCustomTechnique())
```

## Library Architecture

```
promptengineer/
├── __init__.py                 # Main API
├── core/
│   ├── prompt_generator.py     # Main prompt generation engine
│   ├── judge_generator.py      # LLM-as-a-judge prompt generation
│   └── contextual_bandit.py    # Optimization system
├── techniques/
│   ├── base.py                 # Base classes and interfaces
│   ├── chain_of_thought.py     # Chain of Thought implementation
│   ├── chain_of_thoughtlessness.py  # Chain of Thoughtlessness implementation
│   ├── chain_of_draft.py       # Placeholder for future implementation
│   └── registry.py             # Technique management
└── examples/
    └── medical_wound_care.py    # Example usage
```

## Available Techniques

### Chain of Thought
Encourages step-by-step reasoning and explicit thinking processes.

**Status**: ✅ Fully implemented  
**Best for**: Complex reasoning tasks, multi-step problems, mathematical calculations

### Chain of Thoughtlessness  
**Status**: ✅ Fully implemented  
**Based on**: "Chain of Thoughtlessness? An Analysis of CoT in Planning" by Stechly et al. (2024)  
**Best for**: Tasks where direct pattern matching is more effective than verbose reasoning  
**Description**: Emphasizes direct, concise responses over step-by-step reasoning, based on research showing CoT limitations

### Chain of Draft
**Status**: ✅ Implemented  
**Best for**: Tasks requiring efficiency and minimal token usage while maintaining accuracy  
**Description**: Generates minimalistic, draft-like reasoning steps inspired by human cognitive processes. Uses only essential information (≤5 words per step) and symbolic representations to achieve similar accuracy as Chain of Thought while using up to 92% fewer tokens.

## Contextual Features

The bandit system automatically extracts features from your task context:
- **Domain**: Task domain (medical, legal, technical, etc.)
- **Task Complexity**: Estimated complexity based on description
- **Examples**: Whether examples are provided
- **Constraints**: Number and type of constraints
- **Target Audience**: Intended audience type
- **Task Type**: Classification, generation, reasoning, extraction

## Integration with Existing Systems

The library is designed to integrate with existing LLM-as-a-Judge systems:

```python
# Load your existing prompts
existing_prompts = load_your_prompts()

# Enhance them with prompt engineering techniques
for prompt_id, original_prompt in existing_prompts.items():
    context = create_context_from_prompt(original_prompt)
    enhanced_prompt = generator.generate_prompt(context, auto_select=True)
    
    # Use with your existing evaluation pipeline
    evaluation_result = your_evaluation_system(enhanced_prompt.prompt)
    
    # Provide feedback for optimization
    bandit.update_reward(evaluation_result.score)
```

## Performance Tracking

```python
# Get performance summary
summary = generator.get_performance_summary()
print(f"Best technique: {summary['best_technique']}")
print(f"Average scores: {summary['technique_performance']}")

# Contextual performance analysis
bandit_summary = bandit.get_performance_summary()
print("Best techniques by context:")
for context, technique in bandit_summary['best_techniques_by_context'].items():
    print(f"  {context}: {technique}")
```

## Contributing

To add a new prompt engineering technique:

1. Create a new file in `promptengineer/techniques/`
2. Inherit from `BaseTechnique`
3. Implement `get_meta_prompt()` and `get_judge_meta_prompt()` methods
4. Register in `TechniqueRegistry`

## Research Integration

This library is designed to easily integrate new research findings. **Chain of Thoughtlessness** has been successfully implemented based on the Stechly et al. (2024) paper, demonstrating the integration process:

### Successfully Integrated: Chain of Thoughtlessness
- ✅ **Paper**: "Chain of Thoughtlessness? An Analysis of CoT in Planning" by Stechly et al. (2024)
- ✅ **Key Insight**: CoT effectiveness comes from pattern matching, not true algorithmic reasoning
- ✅ **Implementation**: Direct, concise approach avoiding verbose step-by-step reasoning
- ✅ **Evaluation**: Custom criteria focusing on directness and pattern application rather than reasoning explanation

### Successfully Integrated: Chain of Draft
- ✅ **Paper**: "Chain of Draft: Thinking Faster by Writing Less" by Xu et al. (2025)
- ✅ **Key Insight**: Humans use concise drafts/shorthand notes for problem-solving, not verbose explanations
- ✅ **Implementation**: Minimalistic reasoning steps (≤5 words each) focusing on essential information only
- ✅ **Evaluation**: Criteria emphasizing conciseness, symbolic representation, and information density
- ✅ **Performance**: Achieves similar accuracy to CoT while using as little as 7.6% of the tokens

### Integration Process for Future Papers
When you provide research papers, we can:

1. Analyze the core technique principles (as done with Chain of Thoughtlessness)
2. Create meta prompts that capture the essence of the technique  
3. Implement evaluation criteria specific to the technique
4. Add the technique to the registry for optimization
5. Update contextual bandit to learn when to use the new technique

## Example: Medical Wound Care

See `examples/medical_wound_care.py` for a complete demonstration using medical wound care data, showing:
- Loading existing prompts
- Generating Chain of Thought prompts
- Generating Chain of Thoughtlessness prompts
- Generating Chain of Draft prompts (new!)
- Creating LLM-as-a-judge evaluation prompts for all techniques
- Using contextual bandits for optimization
- Complete end-to-end workflow with technique comparison

## License

[Add your license here]

## Citation

If you use this library in your research, please cite:

```bibtex
@software{promptengineer2024,
  title={PromptEngineer: A Modular Library for Automated Prompt Engineering},
  author={[Harivallabha Rangarajan, Shreya Bali]},
  year={2024},
  url={[Repository URL]}
}
``` 