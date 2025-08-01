"""
Example usage of PromptEngineer library with medical wound-care data.

This example demonstrates how to:
1. Load existing prompts from the medical domain
2. Use the library to generate improved prompts with multiple techniques:
   - Chain of Thought (step-by-step reasoning)
   - Chain of Thoughtlessness (direct pattern-based responses)
   - Chain of Draft (minimalistic, draft-like reasoning)
3. Create LLM-as-a-judge evaluation prompts
4. Use contextual bandits for optimization
"""

import pandas as pd
import sys
import os
from typing import Dict, Any

# Add the parent directory to path to import promptengineer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from promptengineer import PromptGenerator, JudgeGenerator, ContextualBandit
from promptengineer.techniques.base import PromptContext

# Import API key from config
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "configs"))
from config import OPENAI_API_KEY


def load_medical_prompts(csv_path: str = "data/prompts.csv") -> Dict[str, str]:
    """Load the existing medical prompts from CSV."""
    try:
        df = pd.read_csv(csv_path)
        prompts = {}
        for _, row in df.iterrows():
            prompts[row['Model']] = row['Prompt']
        return prompts
    except FileNotFoundError:
        # Return sample data if file not found
        return {
            "model_1": "Sample medical wound care prompt for basic dialogue...",
            "model_2": "Sample medical wound care prompt with 14-step procedure...",
            "model_3": "Sample medical wound care prompt with procedure and handout..."
        }


def create_medical_context() -> PromptContext:
    """Create a PromptContext for medical wound care tasks."""
    return PromptContext(
        task_description="""Guide patients through a 14-step wound care procedure while they perform the steps on a fake wound. 
        The model cannot see the patient but must provide appropriate guidance based on patient questions and statements. 
        The model should be thorough, accurate, and safety-focused in medical guidance.""",
        domain="medical",
        examples=[
            {
                "patient_input": "I'm having trouble removing the gauze, it seems stuck.",
                "expected_response": "Gently moisten the gauze with warm water to help loosen it. Don't pull forcefully as this could damage the healing tissue."
            },
            {
                "patient_input": "How much Vaseline should I apply?",
                "expected_response": "Apply a thick layer of Vaseline like you're putting icing on a cupcake. Use more than you think you need to ensure proper wound protection."
            }
        ],
        constraints=[
            "Cannot see visual elements",
            "Must provide safe medical guidance",
            "Should follow the 14-step procedure",
            "Must be clear and non-technical for patients",
            "Should not advance steps prematurely"
        ],
        target_audience="patients performing wound care",
        success_criteria=[
            "Provides accurate medical guidance",
            "Maintains patient safety",
            "Follows appropriate step sequence",
            "Responds appropriately to patient questions",
            "Does not claim visual capabilities"
        ]
    )


def demonstrate_prompt_generation(api_key: str):
    """Demonstrate basic prompt generation."""
    print("=== PromptEngineer Demo: Medical Wound Care ===\n")
    
    # Initialize the prompt generator
    generator = PromptGenerator(api_key=api_key)
    
    # Create context for medical wound care
    context = create_medical_context()
    
    print("1. Available techniques:")
    techniques = generator.list_available_techniques()
    for technique in techniques:
        print(f"   - {technique}")
    print()
    
    # Generate a Chain of Thought prompt
    print("2. Generating Chain of Thought prompt...")
    cot_prompt = generator.generate_prompt(context, technique="chain_of_thought")
    
    print(f"Generated prompt (using {cot_prompt.technique}):")
    print("-" * 50)
    print(cot_prompt.prompt[:500] + "..." if len(cot_prompt.prompt) > 500 else cot_prompt.prompt)
    print("-" * 50)
    print()
    
    # Also generate a Chain of Thoughtlessness prompt for comparison
    print("2b. Generating Chain of Thoughtlessness prompt for comparison...")
    thoughtlessness_prompt = generator.generate_prompt(context, technique="chain_of_thoughtlessness")
    
    print(f"Generated prompt (using {thoughtlessness_prompt.technique}):")
    print("-" * 50)
    print(thoughtlessness_prompt.prompt[:500] + "..." if len(thoughtlessness_prompt.prompt) > 500 else thoughtlessness_prompt.prompt)
    print("-" * 50)
    print()
    
    # Also generate a Chain of Draft prompt for comparison
    print("2c. Generating Chain of Draft prompt for comparison...")
    draft_prompt = generator.generate_prompt(context, technique="chain_of_draft")
    
    print(f"Generated prompt (using {draft_prompt.technique}):")
    print("-" * 50)
    print(draft_prompt.prompt[:500] + "..." if len(draft_prompt.prompt) > 500 else draft_prompt.prompt)
    print("-" * 50)
    print()
    
    return cot_prompt


def demonstrate_judge_generation(api_key: str, cot_prompt):
    """Demonstrate LLM-as-a-judge prompt generation."""
    print("3. Generating LLM-as-a-Judge evaluation prompt...")
    
    # Initialize judge generator
    judge_generator = JudgeGenerator(api_key=api_key)
    
    # Create context for evaluation
    context = create_medical_context()
    
    # Generate judge prompt for Chain of Thought
    judge_prompt = judge_generator.generate_judge_prompt(
        context=context,
        technique_to_evaluate="chain_of_thought",
        scoring_method="pass_fail"
    )
    
    print("Generated judge prompt:")
    print("-" * 50)
    print(judge_prompt.prompt[:500] + "..." if len(judge_prompt.prompt) > 500 else judge_prompt.prompt)
    print("-" * 50)
    print()
    
    print("Evaluation criteria:")
    for i, criterion in enumerate(judge_prompt.criteria, 1):
        print(f"   {i}. {criterion}")
    print()
    
    return judge_prompt


def demonstrate_contextual_bandit():
    """Demonstrate contextual bandit optimization."""
    print("4. Demonstrating Contextual Bandit optimization...")
    
    # Initialize contextual bandit
    bandit = ContextualBandit(epsilon=0.3)  # Higher exploration for demo
    
    # Create context
    context = create_medical_context()
    
    # Simulate several technique selections and rewards
    print("Simulating technique selection and feedback...")
    
    scenarios = [
        ("Simple question about gauze removal", 0.8),
        ("Complex multi-step reasoning about infection signs", 0.9),
        ("Basic procedural guidance", 0.7),
        ("Safety-critical wound assessment", 0.85),
        ("Patient confusion about step sequence", 0.75),
        ("Direct yes/no medical question", 0.95),  # Likely better for Chain of Thoughtlessness
        ("Quick pattern recognition task", 0.9)     # Likely better for Chain of Thoughtlessness
    ]
    
    for scenario, reward in scenarios:
        # Modify context for each scenario
        scenario_context = PromptContext(
            task_description=context.task_description + f" Scenario: {scenario}",
            domain=context.domain,
            examples=context.examples,
            constraints=context.constraints,
            target_audience=context.target_audience,
            success_criteria=context.success_criteria
        )
        
        # Select technique
        selected_technique = bandit.select_technique(scenario_context)
        print(f"   Scenario: {scenario}")
        print(f"   Selected: {selected_technique}")
        
        # Provide feedback
        bandit.update_reward(reward)
        print(f"   Reward: {reward}")
        print()
    
    # Show performance summary
    print("Performance Summary:")
    summary = bandit.get_performance_summary()
    print(f"   Total actions: {summary['total_actions']}")
    print(f"   Current exploration rate: {summary['current_epsilon']:.3f}")
    print("   Technique performance:")
    for technique, performance in summary['technique_performance'].items():
        print(f"      {technique}: {performance['average_reward']:.3f} (n={performance['total_uses']})")
    print()


def demonstrate_end_to_end_workflow(api_key: str):
    """Demonstrate complete workflow."""
    print("5. Complete workflow demonstration...")
    
    # Load existing prompts
    existing_prompts = load_medical_prompts()
    print(f"Loaded {len(existing_prompts)} existing prompts")
    
    # Initialize components
    generator = PromptGenerator(api_key=api_key)
    judge_generator = JudgeGenerator(api_key=api_key)
    bandit = ContextualBandit()
    
    # Create context
    context = create_medical_context()
    
    # Use bandit to select technique
    selected_technique = bandit.select_technique(context)
    print(f"Bandit selected technique: {selected_technique}")
    
    # Generate improved prompt
    improved_prompt = generator.generate_prompt(context, technique=selected_technique)
    print("Generated improved prompt with", improved_prompt.technique)
    
    # Generate judge prompt
    judge_prompt = judge_generator.generate_judge_prompt(
        context=context,
        technique_to_evaluate=selected_technique
    )
    print("Generated evaluation prompt for", judge_prompt.technique_evaluated)
    
    # Simulate evaluation and feedback
    simulated_evaluation_score = 0.85  # This would come from actual evaluation
    bandit.update_reward(simulated_evaluation_score)
    print(f"Updated bandit with reward: {simulated_evaluation_score}")
    
    print("\nWorkflow complete! The system has:")
    print("1. ✓ Selected optimal technique using contextual bandit")
    print("2. ✓ Generated improved prompt using selected technique") 
    print("3. ✓ Created corresponding LLM-as-a-judge evaluation prompt")
    print("4. ✓ Updated optimization system with performance feedback")


def main():
    """Main demonstration function."""
    # Use API key from config
    api_key = OPENAI_API_KEY
    
    if not api_key or api_key == "your-api-key-here":
        print("Please set your OpenAI API key in configs/config.py to run this demo")
        return
    
    try:
        # Run demonstrations
        cot_prompt = demonstrate_prompt_generation(api_key)
        judge_prompt = demonstrate_judge_generation(api_key, cot_prompt)
        demonstrate_contextual_bandit()
        demonstrate_end_to_end_workflow(api_key)
        
        print("=== Demo Complete ===")
        print("\nNext steps:")
        print("- Integrate with your existing evaluation pipeline")
        print("- Add more techniques by implementing the BaseTechnique interface")
        print("- Use the contextual bandit to optimize technique selection over time")
        print("- Customize meta-prompts for your specific domain")
        
    except Exception as e:
        print(f"Demo error: {e}")
        print("This is likely due to API key issues or network connectivity")


if __name__ == "__main__":
    main() 