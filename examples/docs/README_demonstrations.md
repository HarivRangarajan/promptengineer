# Learnable Demonstration Generator with Human-in-the-Loop Feedback

## Overview

The Demonstration Generator is a powerful feature of the PromptEngineer library that automatically generates, refines, and improves demonstrations to clarify task instructions for LLMs. It leverages both automated generation via LLM calls and human-in-the-loop feedback to iteratively improve demonstration quality.

## Key Features

### 🎯 Automated Demonstration Generation
- Generates positive (correct) and negative (incorrect) demonstration examples
- Uses meta-prompting to create contextually relevant demonstrations
- Configurable number of demonstrations per category

### 🔄 Error Feedback Integration
- Seamlessly integrates with mohs-llm-as-a-judge error categorization
- Processes error feedback files (JSON, CSV, or text formats)
- Prioritizes error categories based on frequency and severity

### 🎨 Contrastive Learning
- Creates positive/negative demonstration pairs for each error category
- Highlights boundaries between correct and incorrect interpretations
- Helps LLMs understand nuanced distinctions

### 👥 Human-in-the-Loop Workflow
- Presents demonstrations for human review
- Supports accept/reject/edit actions with feedback
- Maintains audit trail of all decisions and revisions

### 🔧 Pipeline Integration
- Compatible with existing PromptEngineer pipeline
- Exports demonstrations in LLM-ready format
- Provides structured output for further processing

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Task Prompt    │    │ Error Feedback   │    │ Human Reviewer  │
│     Files       │    │   (Judge Output) │    │   Interface     │
└─────────┬───────┘    └─────────┬────────┘    └─────────┬───────┘
          │                      │                       │
          ▼                      ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                DemonstrationGenerator                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Initial   │  │   Error     │  │     Human-in-Loop       │  │
│  │   Generation│  │ Integration │  │     Refinement          │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                Final Demonstrations                             │
│  • Positive Examples        • Error-Specific Pairs             │
│  • Negative Examples        • Human-Reviewed Content           │
│  • LLM-Ready Format         • Audit Trail                      │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Basic Usage

```python
from promptengineer import DemonstrationGenerator, FeedbackIntegrator

# Initialize
demo_generator = DemonstrationGenerator(
    api_key="your-openai-api-key",
    model="gpt-4o",
    output_dir="demo_output"
)

# Load task prompt
prompt_data = demo_generator.load_prompt_file("task_prompt.txt")

# Generate initial demonstrations
demos = demo_generator.generate_initial_demonstrations(prompt_data, "task_1")

# Integrate error feedback
error_categories = demo_generator.load_error_feedback("feedback.json")
refined_demos = demo_generator.refine_demonstrations_with_feedback("task_1", error_categories)

# Get final demonstrations
final_output = demo_generator.get_final_demonstrations("task_1")
```

### Integration with mohs-llm-as-a-judge

```python
from promptengineer import FeedbackIntegrator

# Initialize feedback integrator
feedback_integrator = FeedbackIntegrator()

# Extract error analysis from judge output
judge_output_dir = "/path/to/mohs-llm-as-a-judge/outputs/"
error_analyses = feedback_integrator.extract_from_judge_output(judge_output_dir)

# Convert to demonstration feedback format
demo_feedback = feedback_integrator.convert_to_demonstration_feedback(error_analyses)

# Use priority categories for demonstration refinement
priority_categories = demo_feedback['priority_categories']
```

## File Formats

### Task Prompt Files

Support `.txt` and `.md` files with structured or unstructured content:

```text
Task Description: Medical Wound Care Guidance

Instructions:
1. Provide step-by-step guidance
2. Emphasize safety protocols
3. Never assume visual access

Constraints:
- Cannot see wounds or patient actions
- Must not provide medical diagnosis
- Should refer to professionals for serious cases
```

### Error Feedback Files

Support multiple formats from mohs-llm-as-a-judge:

**JSON Format (Summary Report):**
```json
{
  "technique_analysis": {
    "technique_chain_of_thought": {
      "total_responses": 50,
      "pass_rate": 0.78,
      "failure_categories": {
        "d": 8,  // Visual assumptions
        "b": 3,  // Incorrect information
        "f": 2   // Premature advancement
      }
    }
  }
}
```

**CSV Format (Evaluation Results):**
```csv
technique,label,category,reason
chain_of_thought,0,d,"LLM assumed visual knowledge"
chain_of_thought,1,none,"Appropriate response"
```

## Error Categories

The system recognizes standard error categories from mohs-llm-as-a-judge:

| Category | Name | Description | Priority |
|----------|------|-------------|----------|
| A | Irrelevant Response | Completely ignores question | 1 (Low) |
| B | Incorrect Information | Wrong or contradictory info | 3 (Medium) |
| D | Visual Assumptions | Assumes visual knowledge | 5 (High) |
| E | Inappropriate Uncertainty | Wrong "I don't know" responses | 2 (Low-Medium) |
| F | Premature Advancement | Jumps to future steps | 4 (Medium-High) |

## Human Review Process

### Review Interface Data
```python
review_data = demo_generator.present_for_human_review("task_1")
# Returns:
{
    "task_id": "task_1",
    "demonstrations": [
        {
            "id": "demo_id",
            "content": "demonstration text",
            "type": "positive/negative",
            "error_category": "category_name"
        }
    ],
    "review_instructions": "detailed instructions"
}
```

### Feedback Processing
```python
feedback_data = [
    {
        "demo_id": "demo_1",
        "action": "accept",  # or "reject", "edit"
        "feedback": "Clear and helpful demonstration",
        "edited_content": "optional edited version"
    }
]

updated_demos = demo_generator.process_human_feedback("task_1", feedback_data)
```

## Output Formats

### Final Demonstration Output
```python
final_output = {
    "task_id": "medical_task",
    "positive_demonstrations": [...],
    "negative_demonstrations": [...],
    "error_categories_addressed": ["VISUAL_ASSUMPTIONS", "INCORRECT_INFO"],
    "formatted_for_llm": "Ready-to-use demonstration text",
    "ready_for_prompt_integration": True
}
```

### LLM-Ready Format
```text
DEMONSTRATION EXAMPLES:

✅ CORRECT EXAMPLES (Do this):

1. When a patient asks about wound cleaning, provide clear steps:
   "First, wash your hands thoroughly. Then, gently clean around the wound 
   with saline solution as instructed. Can you confirm you have the saline 
   solution ready?"

❌ INCORRECT EXAMPLES (Avoid this):

1. Never assume you can see the wound:
   Avoid: "I can see the wound looks infected"
   Instead: "Please describe what you're observing about the wound"

Follow the patterns shown in the correct examples and avoid the mistakes 
shown in the incorrect examples.
```

## Advanced Features

### Configurable Parameters
```python
demo_generator = DemonstrationGenerator(
    api_key="your-key",
    model="gpt-4o",
    output_dir="custom_output",
    num_positive_demos=3,  # Number of positive examples
    num_negative_demos=2,  # Number of negative examples
)
```

### Statistics and Monitoring
```python
stats = demo_generator.get_statistics()
# Returns:
{
    "total_tasks": 5,
    "total_demonstrations": 25,
    "demonstration_status_counts": {
        "accepted": 15,
        "rejected": 3,
        "edited": 7
    },
    "error_categories_addressed": ["VISUAL_ASSUMPTIONS", "INCORRECT_INFO"]
}
```

### Audit Trail
All actions are logged with timestamps and details:
```json
{
    "timestamp": "2024-01-15T10:30:00",
    "event_type": "initial_generation",
    "task_id": "medical_task",
    "details": {
        "positive_count": 2,
        "negative_count": 2
    }
}
```

## Best Practices

### 1. Task Prompt Design
- Be specific about constraints and limitations
- Include clear success criteria
- Provide context about the domain and audience

### 2. Error Feedback Integration
- Use recent judge output for current error patterns
- Focus on high-priority error categories first
- Regular updates as error patterns evolve

### 3. Human Review Process
- Train reviewers on demonstration quality criteria
- Provide clear acceptance/rejection guidelines
- Document editing rationale for future reference

### 4. Pipeline Integration
- Start with small demonstration sets for testing
- Gradually increase as quality improves
- Monitor LLM performance improvements

## Troubleshooting

### Common Issues

**1. API Rate Limits**
```python
# Use delays between generation calls
import time
time.sleep(1)  # Add delays if hitting rate limits
```

**2. Large Error Feedback Files**
```python
# Process in batches for large judge outputs
error_analyses = feedback_integrator.extract_from_judge_output(
    judge_output_dir, 
    max_files=10  # Process subset
)
```

**3. Inconsistent Demonstration Quality**
```python
# Increase temperature for more diverse negative examples
demo_generator.client.chat.completions.create(
    temperature=0.8,  # Higher for negative examples
    # ...
)
```

### Performance Optimization

- Use caching for repeated prompt processing
- Batch demonstration generation when possible
- Pre-filter error categories by importance
- Implement async processing for large datasets

## Examples

See `demonstration_generator_demo.py` for a complete working example that demonstrates:
- Loading medical wound care prompts
- Integrating with simulated judge feedback
- Processing human review decisions
- Generating final demonstration sets

Run the demo:
```bash
cd promptengineer/examples
python demonstration_generator_demo.py
```

## API Reference

### DemonstrationGenerator

#### `__init__(api_key, model="gpt-4o", output_dir="demonstration_output", num_positive_demos=2, num_negative_demos=2)`
Initialize the demonstration generator.

#### `load_prompt_file(prompt_file_path) -> Dict[str, str]`
Load and parse a prompt file.

#### `generate_initial_demonstrations(prompt_data, task_id) -> DemonstrationSet`
Generate initial demonstration set.

#### `load_error_feedback(feedback_file_path) -> List[ErrorCategory]`
Load error feedback from file.

#### `refine_demonstrations_with_feedback(task_id, error_categories) -> DemonstrationSet`
Refine demonstrations based on error categories.

#### `present_for_human_review(task_id) -> Dict[str, Any]`
Prepare demonstrations for human review.

#### `process_human_feedback(task_id, feedback_data) -> DemonstrationSet`
Process human reviewer feedback.

#### `get_final_demonstrations(task_id) -> Dict[str, Any]`
Get final demonstration set ready for use.

### FeedbackIntegrator

#### `extract_from_judge_output(judge_output_path) -> List[JudgeErrorAnalysis]`
Extract error analysis from judge output.

#### `convert_to_demonstration_feedback(analyses) -> Dict[str, Any]`
Convert judge analysis to demonstration feedback format.

## Contributing

To extend the demonstration generator:

1. **Custom Error Categories**: Add mappings in `FeedbackIntegrator.category_mappings`
2. **New File Formats**: Extend parsing methods in `DemonstrationGenerator`
3. **Alternative LLM Backends**: Modify the client initialization and API calls
4. **Enhanced Review Interfaces**: Build on the review data structure

## License

Part of the PromptEngineer library. See main library license for details. 