"""
Example: Using Seed Demonstrations with DemonstrationGenerator

This example shows how to use seed demonstrations to guide the generation
of new demonstrations. Seed demonstrations act as templates or reference
points that influence the structure, style, and content of generated examples.
"""

import sys
import os
# Add the path to import the config
sys.path.append(os.path.join(os.path.dirname(__file__), '../../mohs-llm-as-a-judge'))

from promptengineer import DemonstrationGenerator, SeedDemonstration
from configs.config import OPENAI_API_KEY

def create_sample_seed_demonstrations():
    """Create some example seed demonstrations for medical guidance."""
    
    # Positive seed demonstration
    positive_seed = SeedDemonstration(
        content="""Patient asks: "Should I remove the old bandage first?"
        
Medical Assistant Response: "Yes, you should carefully remove the old bandage first. Gently peel it away from the edges, and if it sticks, you can dampen it slightly with clean water to help it come off without causing discomfort. Once removed, dispose of it properly and wash your hands before proceeding with the wound care."

Why this is correct: The response directly addresses the patient's question, provides clear step-by-step guidance, emphasizes safety (hand washing), and gives practical tips (dampening if stuck).""",
        type="positive",
        description="Shows direct, practical response to patient question with clear steps and safety emphasis",
        structure_notes="Format: Question -> Direct answer -> Step-by-step guidance -> Safety note -> Practical tip -> Explanation of correctness"
    )
    
    # Negative seed demonstration  
    negative_seed = SeedDemonstration(
        content="""Patient asks: "Should I remove the old bandage first?"
        
Medical Assistant Response: "I can see that your current bandage looks quite dirty and needs to be changed. The wound underneath appears to be healing well, though there's still some minor bleeding visible around the edges."

Why this is incorrect: The response makes visual assumptions ("I can see", "appears", "visible") when the AI cannot actually see anything. It also doesn't directly answer the patient's specific question about bandage removal.""",
        type="negative", 
        description="Shows problematic visual assumptions and failure to directly answer the question",
        structure_notes="Format: Question -> Problematic response with visual claims -> Explanation of what's wrong"
    )
    
    return [positive_seed, negative_seed]

def demonstrate_seed_guided_generation():
    """Demonstrate generating demonstrations with seed guidance."""
    
    print("🌱 SEED DEMONSTRATION EXAMPLE")
    print("=" * 60)
    
    # Note: You'll need to set your actual OpenAI API key here
    api_key = OPENAI_API_KEY
    
    if api_key == "your-openai-api-key-here":
        print("⚠️  Please set your OpenAI API key in the script to run this example")
        return
    
    # Initialize demonstration generator
    demo_generator = DemonstrationGenerator(
        api_key=api_key,
        output_dir="seed_demo_output",
        num_positive_demos=3,  # Total desired (including seeds)
        num_negative_demos=3   # Total desired (including seeds)
    )
    
    # Create sample prompt data
    prompt_data = {
        "task_description": "Provide medical guidance to patients performing wound care procedures",
        "instructions": "Respond to patient questions during wound care with accurate, safe guidance",
        "constraints": "Cannot see visual elements, must be medically accurate, keep responses concise",
        "context": "Medical wound care assistance"
    }
    
    # Create seed demonstrations
    seed_demos = create_sample_seed_demonstrations()
    
    print(f"📋 Created {len(seed_demos)} seed demonstrations:")
    for i, seed in enumerate(seed_demos, 1):
        print(f"   {i}. {seed.type.upper()}: {seed.content[:100]}...")
        if seed.description:
            print(f"      Description: {seed.description}")
        print()
    
    # Generate demonstrations using seeds
    print("🚀 Generating demonstrations with seed guidance...")
    demo_set = demo_generator.generate_demonstrations_with_seeds(
        prompt_data=prompt_data,
        seed_demonstrations=seed_demos,
        task_id="medical_wound_care_with_seeds"
    )
    
    print(f"✅ Generated demonstration set:")
    print(f"   • Task: {demo_set.task_description}")
    print(f"   • Positive demonstrations: {len(demo_set.positive_demonstrations)}")
    print(f"   • Negative demonstrations: {len(demo_set.negative_demonstrations)}")
    print(f"   • Seed demonstrations: {len(demo_set.seed_demonstrations)}")
    print(f"   • Generation method: {demo_set.metadata.get('generation_method')}")
    
    # Show seed summary
    seed_summary = demo_generator.get_seed_demonstrations_summary("medical_wound_care_with_seeds")
    print(f"\n📊 Seed Summary:")
    print(f"   • Total seeds: {seed_summary['total_seeds']}")
    print(f"   • Positive seeds: {seed_summary['positive_seeds']}")
    print(f"   • Negative seeds: {seed_summary['negative_seeds']}")
    print(f"   • Seeds with descriptions: {seed_summary['seeds_with_descriptions']}")
    
    # Display generated demonstrations
    print(f"\n📝 GENERATED DEMONSTRATIONS:")
    print("=" * 60)
    
    print("\n✅ POSITIVE DEMONSTRATIONS:")
    for i, demo in enumerate(demo_set.positive_demonstrations, 1):
        seed_marker = " [SEED]" if demo.is_seed else " [GENERATED]"
        print(f"\n{i}.{seed_marker}")
        print(f"{demo.content}")
        print("-" * 40)
    
    print("\n❌ NEGATIVE DEMONSTRATIONS:")
    for i, demo in enumerate(demo_set.negative_demonstrations, 1):
        seed_marker = " [SEED]" if demo.is_seed else " [GENERATED]"
        print(f"\n{i}.{seed_marker}")
        print(f"{demo.content}")
        print("-" * 40)
    
    # Get final formatted demonstrations
    final_demos = demo_generator.get_final_demonstrations("medical_wound_care_with_seeds")
    print(f"\n🎯 FORMATTED FOR LLM USE:")
    print("=" * 60)
    print(final_demos['formatted_for_llm'])
    
    return demo_set

def demonstrate_different_seed_formats():
    """Show different ways to provide seed demonstrations."""
    
    print("\n🔄 DIFFERENT SEED FORMATS EXAMPLE")
    print("=" * 60)
    
    # Method 1: Create SeedDemonstration objects directly
    seed_objects = [
        SeedDemonstration(
            content="Direct answer with clear steps",
            type="positive",
            description="Concise and helpful"
        )
    ]
    
    # Method 2: Use dictionary format
    seed_dicts = [
        {
            "content": "Response that makes visual assumptions",
            "type": "negative", 
            "description": "Shows what not to do",
            "structure_notes": "Avoid 'I can see' language"
        }
    ]
    
    # Method 3: Simple list format
    seed_simple = {
        "positive_examples": ["Give clear, step-by-step guidance"],
        "negative_examples": ["I can see your wound looks infected"]
    }
    
    print("Seed format examples:")
    print("1. SeedDemonstration objects - Full control")
    print("2. Dictionary format - Structured data")
    print("3. Simple format - Quick setup")
    
    return seed_objects, seed_dicts, seed_simple

if __name__ == "__main__":
    # Run the main demonstration
    demo_set = demonstrate_seed_guided_generation()
    
    # Show different formats
    demonstrate_different_seed_formats()
    
    print(f"\n🎉 Example completed! Check 'seed_demo_output/' for saved files.")
    print(f"💡 Tip: Use seed demonstrations to maintain consistency in style and structure across generated examples.") 