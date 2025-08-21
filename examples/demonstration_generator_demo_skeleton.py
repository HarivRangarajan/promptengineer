#!/usr/bin/env python3
"""
Demonstration Generator with Human-in-the-Loop Feedback - Complete Demo

This script demonstrates the full workflow of the learnable demonstration generator:
1. Loading task prompts and generating initial demonstrations
2. Integrating error feedback from mohs-llm-as-a-judge
3. Refining demonstrations based on error categories
4. Human-in-the-loop review and feedback
5. Generating final demonstrations for prompt engineering

Requirements:
- OpenAI API key set in environment variable OPENAI_API_KEY
- mohs-llm-as-a-judge output files (optional, demo data provided)
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Any

# Add promptengineer to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from promptengineer import DemonstrationGenerator, FeedbackIntegrator


def create_demo_prompt_file() -> str:
    """Create a demo prompt file for medical wound care guidance."""
    prompt_content = """
Task Description: Medical Wound Care Guidance System

You are an AI assistant helping patients learn proper wound care procedures. Your role is to provide clear, safe, and accurate guidance while acknowledging your limitations.

Instructions:
1. Provide step-by-step guidance for wound care procedures
2. Emphasize safety and hygiene at every step
3. Never assume you can see the wound or patient's actions
4. Ask for patient confirmation before proceeding to next steps
5. Refer to medical professionals for serious concerns
6. Use clear, non-technical language when possible

Constraints:
- Cannot see visual elements (wounds, supplies, patient actions)
- Must not provide diagnosis or prescriptions
- Should encourage seeking professional help for serious wounds
- Must emphasize proper hygiene and safety protocols
- Cannot confirm visual details or locations

Context:
This system will be used by patients practicing wound care procedures on training materials. The AI must guide them through proper techniques while maintaining appropriate boundaries regarding visual information and medical limitations.
"""
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    temp_file.write(prompt_content)
    temp_file.close()
    
    return temp_file.name


def create_demo_feedback_file() -> str:
    """Create demo error feedback based on mohs-llm-as-a-judge format."""
    # Simulate realistic error feedback from LLM-as-a-judge evaluation
    feedback_data = {
        "evaluation_type": "prompt_engineering_techniques_vs_official_responses",
        "timestamp": "2024-01-15T10:30:00",
        "model_used": "gpt-4o",
        "techniques_evaluated": ["chain_of_thought", "chain_of_thoughtlessness", "chain_of_draft"],
        "total_responses_evaluated": 150,
        "overall_pass_rate": 0.72,
        "technique_analysis": {
            "technique_chain_of_thought": {
                "total_responses": 50,
                "pass_rate": 0.78,
                "failure_categories": {
                    "d": 8,  # Visual assumptions
                    "b": 3,  # Incorrect information
                    "f": 2   # Premature advancement
                },
                "detailed_failures": [
                    {
                        "category": "d",
                        "reason": "LLM confirmed seeing specific wound characteristics without visual access",
                        "example": "Yes, I can see the wound is healing well"
                    },
                    {
                        "category": "d", 
                        "reason": "LLM indicated visual confirmation of supplies location",
                        "example": "Yes, that's the correct gauze you're holding"
                    },
                    {
                        "category": "b",
                        "reason": "LLM provided incorrect procedural steps contradicting context",
                        "example": "Use the paper towels from the cabinet" 
                    }
                ]
            },
            "technique_chain_of_thoughtlessness": {
                "total_responses": 50,
                "pass_rate": 0.64,
                "failure_categories": {
                    "d": 12,  # Visual assumptions - higher rate
                    "b": 5,   # Incorrect information
                    "a": 1,   # Irrelevant responses
                    "f": 3    # Premature advancement
                },
                "detailed_failures": [
                    {
                        "category": "d",
                        "reason": "LLM assumed visual context repeatedly",
                        "example": "I can see you've placed the bandage correctly"
                    },
                    {
                        "category": "b",
                        "reason": "LLM contradicted provided instructions",
                        "example": "Clean the wound with soap and water instead of saline"
                    }
                ]
            },
            "technique_chain_of_draft": {
                "total_responses": 50,
                "pass_rate": 0.74,
                "failure_categories": {
                    "d": 6,   # Visual assumptions
                    "b": 4,   # Incorrect information
                    "f": 3    # Premature advancement
                },
                "detailed_failures": [
                    {
                        "category": "f",
                        "reason": "LLM provided future step instructions prematurely",
                        "example": "Now apply the final bandage" 
                    }
                ]
            }
        }
    }
    
    # Create temporary JSON file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(feedback_data, temp_file, indent=2)
    temp_file.close()
    
    return temp_file.name


def simulate_human_feedback() -> List[Dict[str, Any]]:
    """Simulate human reviewer feedback on generated demonstrations."""
    # In a real scenario, this would be interactive
    print("\n" + "="*60)
    print("SIMULATING HUMAN REVIEW PROCESS")
    print("="*60)
    print("In a real implementation, human reviewers would:")
    print("1. Review each generated demonstration")
    print("2. Accept, reject, or edit demonstrations") 
    print("3. Provide specific feedback for improvements")
    print("4. Ensure demonstrations address error categories effectively")
    print("\nFor this demo, we're simulating typical reviewer decisions...")
    
    # Simulate realistic human feedback patterns
    feedback_patterns = [
        {
            "action": "accept",
            "feedback": "Clear and accurate demonstration",
            "probability": 0.6
        },
        {
            "action": "edit",
            "feedback": "Good concept but needs clarity improvements",
            "edited_content": "Improved version with clearer instructions and better emphasis on safety protocols.",
            "probability": 0.3
        },
        {
            "action": "reject",
            "feedback": "Unclear or potentially misleading",
            "probability": 0.1
        }
    ]
    
    return feedback_patterns


def interactive_demo():
    """Run the complete demonstration workflow."""
    print("🏥 DEMONSTRATION GENERATOR - MEDICAL WOUND CARE DEMO")
    print("="*60)
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ Error: Please set OPENAI_API_KEY environment variable")
        print("   You can get an API key from: https://platform.openai.com/api-keys")
        return
    
    print("✅ OpenAI API key found")
    
    # Initialize components
    print("\n📝 Initializing Demonstration Generator...")
    demo_generator = DemonstrationGenerator(
        api_key=api_key,
        model="gpt-4o",
        output_dir="demo_output",
        num_positive_demos=2,
        num_negative_demos=2
    )
    
    feedback_integrator = FeedbackIntegrator()
    
    # Step 1: Load task prompt
    print("\n📋 Step 1: Loading task prompt...")
    prompt_file = create_demo_prompt_file()
    print(f"   Created demo prompt file: {prompt_file}")
    
    try:
        prompt_data = demo_generator.load_prompt_file(prompt_file)
        print(f"   ✅ Loaded prompt with task: {prompt_data['task_description'][:80]}...")
    except Exception as e:
        print(f"   ❌ Error loading prompt: {e}")
        return
    
    # Step 2: Generate initial demonstrations
    print("\n🎯 Step 2: Generating initial demonstrations...")
    task_id = "medical_wound_care_demo"
    
    try:
        initial_demos = demo_generator.generate_initial_demonstrations(prompt_data, task_id)
        print(f"   ✅ Generated {len(initial_demos.positive_demonstrations)} positive demonstrations")
        print(f"   ✅ Generated {len(initial_demos.negative_demonstrations)} negative demonstrations")
        
        # Show sample demonstration
        if initial_demos.positive_demonstrations:
            sample_demo = initial_demos.positive_demonstrations[0]
            print(f"\n   📄 Sample Positive Demonstration:")
            print(f"   {sample_demo.content[:200]}...")
            
    except Exception as e:
        print(f"   ❌ Error generating demonstrations: {e}")
        return
    
    # Step 3: Load and process error feedback  
    print("\n📊 Step 3: Processing error feedback from mohs-llm-as-a-judge...")
    feedback_file = create_demo_feedback_file()
    print(f"   Created demo feedback file: {feedback_file}")
    
    try:
        # Extract error analysis
        error_analyses = feedback_integrator.extract_from_judge_output(feedback_file)
        print(f"   ✅ Analyzed {len(error_analyses)} technique evaluations")
        
        # Convert to demonstration feedback format
        demo_feedback = feedback_integrator.convert_to_demonstration_feedback(error_analyses)
        print(f"   ✅ Identified {len(demo_feedback['priority_categories'])} priority error categories")
        
        # Show top error categories
        if demo_feedback['priority_categories']:
            print(f"\n   🎯 Top Error Categories:")
            for i, category in enumerate(demo_feedback['priority_categories'][:3], 1):
                print(f"   {i}. {category['name']}: {category['total_count']} occurrences")
                print(f"      Priority: {category['priority']}, Description: {category['description'][:60]}...")
        
        # Load error categories for demonstration refinement
        error_categories = demo_generator.load_error_feedback(feedback_file)
        print(f"   ✅ Loaded {len(error_categories)} error categories for refinement")
        
    except Exception as e:
        print(f"   ❌ Error processing feedback: {e}")
        return
    
    # Step 4: Refine demonstrations with error feedback
    print("\n🔧 Step 4: Refining demonstrations based on error feedback...")
    
    try:
        refined_demos = demo_generator.refine_demonstrations_with_feedback(task_id, error_categories)
        total_demos = len(refined_demos.positive_demonstrations) + len(refined_demos.negative_demonstrations)
        print(f"   ✅ Refined demonstration set now has {total_demos} total demonstrations")
        
        # Show contrastive example
        error_specific_demos = [d for d in refined_demos.positive_demonstrations if d.error_category]
        if error_specific_demos:
            sample_refined = error_specific_demos[0]
            print(f"\n   📄 Sample Error-Specific Demonstration (Category: {sample_refined.error_category}):")
            print(f"   {sample_refined.content[:200]}...")
            
    except Exception as e:
        print(f"   ❌ Error refining demonstrations: {e}")
        return
    
    # Step 5: Present for human review
    print("\n👥 Step 5: Presenting demonstrations for human review...")
    
    try:
        review_data = demo_generator.present_for_human_review(task_id)
        print(f"   ✅ Prepared {len(review_data['demonstrations'])} demonstrations for review")
        print(f"   📝 Review covers {len(review_data['error_categories'])} error categories")
        
        # Simulate human feedback
        feedback_patterns = simulate_human_feedback()
        
        # Generate simulated feedback for each demonstration
        simulated_feedback = []
        for demo in review_data['demonstrations']:
            # Randomly assign feedback based on patterns
            import random
            pattern = random.choices(
                feedback_patterns,
                weights=[p['probability'] for p in feedback_patterns]
            )[0]
            
            feedback_item = {
                "demo_id": demo['id'],
                "action": pattern['action'],
                "feedback": pattern['feedback']
            }
            
            if pattern['action'] == 'edit' and 'edited_content' in pattern:
                feedback_item['edited_content'] = pattern['edited_content']
            
            simulated_feedback.append(feedback_item)
        
        print(f"   ✅ Simulated human feedback for all demonstrations")
        
    except Exception as e:
        print(f"   ❌ Error in review process: {e}")
        return
    
    # Step 6: Process human feedback
    print("\n✏️  Step 6: Processing human feedback...")
    
    try:
        updated_demos = demo_generator.process_human_feedback(task_id, simulated_feedback)
        
        # Count feedback results
        accepted = len([f for f in simulated_feedback if f['action'] == 'accept'])
        edited = len([f for f in simulated_feedback if f['action'] == 'edit'])
        rejected = len([f for f in simulated_feedback if f['action'] == 'reject'])
        
        print(f"   ✅ Processed feedback: {accepted} accepted, {edited} edited, {rejected} rejected")
        
    except Exception as e:
        print(f"   ❌ Error processing feedback: {e}")
        return
    
    # Step 7: Generate final demonstrations
    print("\n🎉 Step 7: Generating final demonstration set...")
    
    try:
        final_output = demo_generator.get_final_demonstrations(task_id)
        
        total_final = final_output['total_demonstrations']
        positive_count = len(final_output['positive_demonstrations'])
        negative_count = len(final_output['negative_demonstrations'])
        
        print(f"   ✅ Final set contains {total_final} demonstrations")
        print(f"   📊 Breakdown: {positive_count} positive, {negative_count} negative")
        print(f"   🎯 Addresses {len(final_output['error_categories_addressed'])} error categories")
        
        # Show formatted output for LLM integration
        print(f"\n   📋 Formatted for LLM Integration:")
        formatted_demo = final_output['formatted_for_llm'][:300]
        print(f"   {formatted_demo}...")
        
    except Exception as e:
        print(f"   ❌ Error generating final output: {e}")
        return
    
    # Step 8: Show statistics and summary
    print("\n📈 Step 8: Generation Statistics...")
    
    try:
        stats = demo_generator.get_statistics()
        print(f"   📊 Total tasks processed: {stats['total_tasks']}")
        print(f"   📝 Total demonstrations generated: {stats['total_demonstrations']}")
        print(f"   ✅ Status breakdown:")
        for status, count in stats['demonstration_status_counts'].items():
            print(f"       {status.title()}: {count}")
        print(f"   🎯 Error categories addressed: {len(stats['error_categories_addressed'])}")
        print(f"   📋 Audit events logged: {stats['audit_events']}")
        
    except Exception as e:
        print(f"   ❌ Error getting statistics: {e}")
    
    # Final summary
    print("\n" + "="*60)
    print("🎉 DEMONSTRATION GENERATOR DEMO COMPLETE!")
    print("="*60)
    print(f"✅ Successfully demonstrated the complete workflow:")
    print(f"   1. ✅ Loaded task prompt and generated initial demonstrations")
    print(f"   2. ✅ Integrated error feedback from mohs-llm-as-a-judge")
    print(f"   3. ✅ Refined demonstrations based on error categories")
    print(f"   4. ✅ Simulated human review and feedback process")
    print(f"   5. ✅ Generated final demonstration set for prompt engineering")
    
    print(f"\n📁 Output files saved to: demo_output/")
    print(f"   • Demonstration sets: demo_output/demonstrations/")
    print(f"   • Audit trails: demo_output/audit_trails/")
    print(f"   • Final output: demo_output/final_demonstrations_{task_id}.json")
    
    print(f"\n🔬 Key Benefits Demonstrated:")
    print(f"   • Automated demonstration generation with LLM")
    print(f"   • Error-driven refinement using judge feedback")
    print(f"   • Human-in-the-loop quality control")
    print(f"   • Contrastive learning (positive/negative examples)")
    print(f"   • Seamless integration with existing prompt engineering pipeline")
    
    print(f"\n🚀 Next Steps:")
    print(f"   • Integrate with your existing prompt engineering workflow")
    print(f"   • Use real mohs-llm-as-a-judge output files")
    print(f"   • Implement interactive human review interface")
    print(f"   • Scale to multiple tasks and domains")
    
    # Cleanup
    os.unlink(prompt_file)
    os.unlink(feedback_file)
    print(f"\n🧹 Cleaned up temporary demo files")


def show_integration_example():
    """Show how to integrate with existing prompt engineering pipeline."""
    print("\n" + "="*60)
    print("INTEGRATION WITH EXISTING PIPELINE EXAMPLE")
    print("="*60)
    
    integration_code = '''
# Example: Integrating with existing PromptEngineer pipeline

from promptengineer import PromptGenerator, DemonstrationGenerator, FeedbackIntegrator

def enhanced_prompt_engineering_with_demonstrations():
    """Enhanced prompt engineering with learnable demonstrations."""
    
    # Initialize components
    prompt_generator = PromptGenerator(api_key="your-api-key")
    demo_generator = DemonstrationGenerator(api_key="your-api-key")
    feedback_integrator = FeedbackIntegrator()
    
    # Step 1: Generate base prompt using existing technique
    context = PromptContext(
        task_description="Medical guidance system",
        domain="healthcare", 
        constraints=["No visual access", "Safety critical"]
    )
    
    base_prompt = prompt_generator.generate_prompt(context, technique="chain_of_thought")
    
    # Step 2: Generate demonstrations for the prompt
    prompt_data = demo_generator.load_prompt_file("task_prompt.txt")
    demonstrations = demo_generator.generate_initial_demonstrations(prompt_data, "medical_task")
    
    # Step 3: Integrate error feedback from judge system
    judge_output = "/path/to/mohs-llm-as-a-judge/outputs/"
    error_analyses = feedback_integrator.extract_from_judge_output(judge_output)
    error_categories = demo_generator.load_error_feedback("feedback.json")
    
    # Step 4: Refine demonstrations based on errors
    refined_demos = demo_generator.refine_demonstrations_with_feedback(
        "medical_task", error_categories
    )
    
    # Step 5: Get final demonstrations for prompt integration
    final_demos = demo_generator.get_final_demonstrations("medical_task")
    
    # Step 6: Create enhanced prompt with demonstrations
    enhanced_prompt = f\"\"\"
{base_prompt.prompt}

{final_demos['formatted_for_llm']}

Now apply these guidelines to respond to the user query.
\"\"\"
    
    return enhanced_prompt

# Usage in production pipeline
enhanced_prompt = enhanced_prompt_engineering_with_demonstrations()
# Use enhanced_prompt with your LLM for improved performance
'''
    
    print(integration_code)


if __name__ == "__main__":
    print("🚀 Starting Demonstration Generator Demo...")
    
    try:
        # Run main interactive demo
        interactive_demo()
        
        # Show integration example
        show_integration_example()
        
    except KeyboardInterrupt:
        print("\n\n⚡ Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n👋 Demo finished. Thank you!") 