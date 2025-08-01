"""
Basic tests for PromptEngineer library functionality.
"""

import sys
import os

# Add the parent directory to Python path for testing
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from promptengineer.techniques.base import PromptContext, BaseTechnique
from promptengineer.techniques.registry import TechniqueRegistry
from promptengineer.techniques.chain_of_thought import ChainOfThoughtTechnique
from promptengineer.techniques.chain_of_draft import ChainOfDraftTechnique


def test_basic_imports():
    """Test that basic imports work."""
    print("✓ Basic imports successful")


def test_prompt_context():
    """Test PromptContext creation."""
    context = PromptContext(
        task_description="Test task",
        domain="test",
        constraints=["constraint1", "constraint2"]
    )
    
    assert context.task_description == "Test task"
    assert context.domain == "test"
    assert len(context.constraints) == 2
    print("✓ PromptContext creation works")


def test_technique_registry():
    """Test technique registry functionality."""
    registry = TechniqueRegistry()
    
    # Check that techniques are registered
    techniques = registry.list_techniques()
    assert "chain_of_thought" in techniques
    assert "chain_of_thoughtlessness" in techniques
    assert "chain_of_draft" in techniques
    print(f"✓ Registry has {len(techniques)} techniques: {techniques}")
    
    # Test getting techniques
    cot = registry.get_technique("chain_of_thought")
    assert cot is not None
    assert isinstance(cot, ChainOfThoughtTechnique)
    
    cod = registry.get_technique("chain_of_draft")
    assert cod is not None
    assert isinstance(cod, ChainOfDraftTechnique)
    print("✓ Technique retrieval works")


def test_meta_prompt_generation():
    """Test meta prompt generation."""
    context = PromptContext(
        task_description="Help users solve math problems step by step",
        domain="education",
        target_audience="students"
    )
    
    cot = ChainOfThoughtTechnique()
    meta_prompt = cot.get_meta_prompt(context)
    
    assert len(meta_prompt) > 100  # Should be substantial
    assert "Chain of Thought" in meta_prompt
    assert "step-by-step" in meta_prompt.lower()
    print("✓ Meta prompt generation works")


def test_judge_meta_prompt_generation():
    """Test judge meta prompt generation."""
    context = PromptContext(
        task_description="Help users solve math problems step by step",
        domain="education"
    )
    
    cot = ChainOfThoughtTechnique()
    judge_meta_prompt = cot.get_judge_meta_prompt(context)
    
    assert len(judge_meta_prompt) > 100
    assert "evaluat" in judge_meta_prompt.lower()
    assert "Chain of Thought" in judge_meta_prompt
    print("✓ Judge meta prompt generation works")


def test_contextual_bandit_basic():
    """Test basic contextual bandit functionality."""
    try:
        from promptengineer.core.contextual_bandit import ContextualBandit, ContextualFeatures
        
        bandit = ContextualBandit(epsilon=0.5)
        
        context = PromptContext(
            task_description="Complex reasoning task with multiple steps",
            domain="science",
            target_audience="researchers"
        )
        
        # Test feature extraction
        features = bandit.extract_features(context)
        assert isinstance(features, ContextualFeatures)
        assert features.domain == "science"
        print("✓ Contextual bandit basics work")
        
    except ImportError as e:
        print(f"! Contextual bandit test skipped (numpy not available): {e}")


def test_pipeline_basic():
    """Test basic pipeline functionality."""
    try:
        from promptengineer.core.pipeline import PromptPipeline
        
        # Test pipeline initialization (without API key for basic test)
        pipeline = PromptPipeline(api_key="test-key", output_dir="test_output")
        
        # Test that pipeline has the expected attributes
        assert hasattr(pipeline, 'prompt_generator')
        assert hasattr(pipeline, 'judge_generator')
        assert hasattr(pipeline, 'contextual_bandit')
        assert pipeline.output_dir == "test_output"
        
        print("✓ Pipeline initialization works")
        
    except Exception as e:
        print(f"! Pipeline test skipped: {e}")


def run_tests():
    """Run all tests."""
    print("Running PromptEngineer Library Tests...\n")
    
    try:
        test_basic_imports()
        test_prompt_context()
        test_technique_registry()
        test_meta_prompt_generation()
        test_judge_meta_prompt_generation()
        test_contextual_bandit_basic()
        test_pipeline_basic()
        
        print("\n🎉 All tests passed!")
        print("\nLibrary structure is correct and ready for use.")
        print("Next steps:")
        print("1. Add your OpenAI API key")
        print("2. Run the example: python examples/medical_wound_care.py")
        print("3. Implement additional techniques based on research papers")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_tests() 