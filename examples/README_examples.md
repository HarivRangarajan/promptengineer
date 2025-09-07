# PromptEngineer Examples

This directory contains examples of how to use the PromptEngineer library.

## Data Files

The examples require data files that are not included in this repository. You should provide your own data files in the `examples/data/` directory.

### Required Data Files

For the medical wound care example (`medical_wound_care.py`), you need:

1. `medical_queries_dataset.csv` - Contains patient queries with dialogue context
2. `prompts.csv` - Contains existing medical prompts (optional)

### Data Format

The `medical_queries_dataset.csv` should have the following columns:
- `query`: The patient's question
- `dialogue_context`: The researcher's instruction
- `context`: Procedural context
- `step`: Current step in the procedure
- `participant_id`: Participant identifier

### Getting Started

1. Create the `examples/data/` directory:
   ```bash
   mkdir -p examples/data
   ```

2. Add your data files to the `examples/data/` directory

3. Run the examples:
   ```bash
   python examples/medical_wound_care.py
   ```

## Example Structure

```
examples/
├── README.md
├── __init__.py
├── data/                    # Add your data files here
│   ├── medical_queries_dataset.csv
│   └── prompts.csv
└── medical_wound_care.py    # Medical wound care pipeline example
```

## Installation

Before running examples, install the PromptEngineer library:

```bash
pip install -e .
```

## Configuration

Set your OpenAI API key in your environment or create a config file:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Available Examples

- **medical_wound_care.py**: Complete medical wound care pipeline with contextual bandit optimization
  - Demonstrates prompt generation, evaluation, and optimization
  - Uses real medical dialogue data
  - Shows LLM-as-a-Judge evaluation capabilities 