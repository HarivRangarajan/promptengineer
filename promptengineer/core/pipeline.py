"""
End-to-end pipeline for prompt engineering, evaluation, and optimization.
"""

import json
import os
import csv
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd

from .prompt_generator import PromptGenerator
from .judge_generator import JudgeGenerator
from .contextual_bandit import ContextualBandit
from ..techniques.base import PromptContext


class PromptPipeline:
    """
    Complete pipeline for prompt engineering, dataset processing, evaluation, and optimization.
    
    This class manages:
    1. Loading queries/datasets
    2. Generating enhanced prompts using different techniques
    3. Running prompts on datasets to generate responses
    4. Evaluating responses with LLM-as-a-judge
    5. Optimizing technique selection with contextual bandit
    6. Saving all artifacts for analysis
    """
    
    def __init__(self, api_key: str, output_dir: str = "prompt_pipeline_output"):
        """
        Initialize the pipeline.
        
        Args:
            api_key: OpenAI API key
            output_dir: Directory to save all pipeline outputs
        """
        self.api_key = api_key
        self.output_dir = output_dir
        
        # Initialize core components
        self.prompt_generator = PromptGenerator(api_key=api_key)
        self.judge_generator = JudgeGenerator(api_key=api_key)
        self.contextual_bandit = ContextualBandit(epsilon=0.1)
        
        # Create output directory structure
        self._setup_output_directories()
        
        # Pipeline state
        self.generated_prompts = {}
        self.judge_prompts = {}
        self.responses = {}
        self.evaluations = {}
        self.performance_history = []
    
    def _setup_output_directories(self):
        """Create the output directory structure."""
        subdirs = [
            "prompts",
            "judge_prompts", 
            "responses",
            "evaluations",
            "datasets",
            "reports"
        ]
        
        os.makedirs(self.output_dir, exist_ok=True)
        for subdir in subdirs:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)
    
    def load_dataset(self, dataset_path: str, query_column: str = "query", 
                     context_column: str = "context") -> List[Dict]:
        """Load a dataset of queries to process."""
        print(f"Loading dataset from: {dataset_path}")
        
        if dataset_path.endswith('.csv'):
            df = pd.read_csv(dataset_path)
            queries = []
            for _, row in df.iterrows():
                query_dict = {"query": row[query_column]}
                
                # Always try to include context if the column exists
                if context_column and context_column in df.columns:
                    query_dict["context"] = str(row[context_column]) if pd.notna(row[context_column]) else ""
                else:
                    query_dict["context"] = ""
                
                # Include any additional columns that might be useful
                for col in df.columns:
                    if col not in [query_column, context_column]:
                        query_dict[col] = row[col]
                        
                queries.append(query_dict)
        else:
            raise ValueError("Dataset must be CSV format")
        
        print(f"Loaded {len(queries)} queries with context column: {context_column}")
        return queries
    
    def generate_enhanced_prompts(self, context: PromptContext, techniques: List[str] = None) -> Dict[str, Any]:
        """Generate enhanced prompts using different techniques."""
        if techniques is None:
            techniques = self.prompt_generator.list_available_techniques()
        
        print(f"Generating prompts using techniques: {techniques}")
        
        enhanced_prompts = {}
        for technique in techniques:
            print(f"  Generating {technique} prompt...")
            
            try:
                generated_prompt = self.prompt_generator.generate_prompt(
                    context=context,
                    technique=technique
                )
                enhanced_prompts[technique] = {
                    "prompt": generated_prompt.prompt,
                    "technique": generated_prompt.technique,
                    "meta_prompt_used": generated_prompt.meta_prompt_used,
                    "timestamp": datetime.now().isoformat()
                }
                print(f"    ✓ Generated ({len(generated_prompt.prompt)} chars)")
                
            except Exception as e:
                print(f"    ✗ Error generating {technique}: {e}")
        
        # Save generated prompts
        self.generated_prompts.update(enhanced_prompts)
        self._save_prompts(enhanced_prompts)
        
        return enhanced_prompts
    
    def generate_judge_prompts(self, context: PromptContext, techniques: List[str] = None,
                              scoring_method: str = "rubric") -> Dict[str, Any]:
        """Generate LLM-as-a-judge evaluation prompts for each technique."""
        if techniques is None:
            techniques = list(self.generated_prompts.keys())
        
        print(f"Generating judge prompts for techniques: {techniques}")
        
        judge_prompts = {}
        for technique in techniques:
            print(f"  Generating judge prompt for {technique}...")
            
            try:
                judge_prompt = self.judge_generator.generate_judge_prompt(
                    context=context,
                    technique_to_evaluate=technique,
                    scoring_method=scoring_method
                )
                
                judge_prompts[technique] = {
                    "prompt": judge_prompt.prompt,
                    "criteria": judge_prompt.criteria,
                    "scoring_method": judge_prompt.scoring_method,
                    "technique_evaluated": judge_prompt.technique_evaluated,
                    "meta_prompt_used": judge_prompt.meta_prompt_used,
                    "timestamp": datetime.now().isoformat()
                }
                print(f"    ✓ Generated judge prompt ({len(judge_prompt.criteria)} criteria)")
                
            except Exception as e:
                print(f"    ✗ Error generating judge prompt for {technique}: {e}")
        
        # Save judge prompts
        self.judge_prompts.update(judge_prompts)
        self._save_judge_prompts(judge_prompts)
        
        return judge_prompts
    
    def run_prompts_on_dataset(self, queries: List[Dict], techniques: List[str] = None) -> Dict[str, List[Dict]]:
        """Run generated prompts on the dataset to produce responses."""
        if techniques is None:
            techniques = list(self.generated_prompts.keys())
        
        print(f"Running prompts on {len(queries)} queries using {len(techniques)} techniques...")
        
        all_responses = {}
        
        for technique in techniques:
            if technique not in self.generated_prompts:
                print(f"  Warning: No generated prompt for {technique}, skipping")
                continue
            
            print(f"  Running {technique} on dataset...")
            technique_responses = []
            
            prompt_template = self.generated_prompts[technique]["prompt"]
            
            for i, query_item in enumerate(queries):
                query = query_item["query"]
                context = query_item.get("context", "")
                
                # Combine prompt template with the specific query
                full_prompt = f"{prompt_template}\n\nQuery: {query}"
                if context:
                    full_prompt += f"\nContext: {context}"
                
                try:
                    # Make LLM call to generate response
                    api_response = self.prompt_generator.client.chat.completions.create(
                        model=self.prompt_generator.model,
                        messages=[
                            {"role": "system", "content": "You are a helpful medical assistant providing wound care guidance."},
                            {"role": "user", "content": full_prompt}
                        ],
                        temperature=self.prompt_generator.temperature
                    )
                    response = api_response.choices[0].message.content
                    
                    response_dict = {
                        "query": query,
                        "context": context,
                        "prompt_used": prompt_template,
                        "full_prompt": full_prompt,
                        "response": response,
                        "technique": technique,
                        "timestamp": datetime.now().isoformat(),
                        "query_index": i
                    }
                    
                    # Include any additional metadata from the original query item
                    for key, value in query_item.items():
                        if key not in ["query", "context"]:
                            response_dict[key] = value
                    
                    technique_responses.append(response_dict)
                    print(f"    Query {i+1}/{len(queries)} completed")
                    
                except Exception as e:
                    print(f"    Error processing query {i+1}: {e}")
                    error_dict = {
                        "query": query,
                        "context": context,
                        "error": str(e),
                        "technique": technique,
                        "timestamp": datetime.now().isoformat(),
                        "query_index": i
                    }
                    
                    # Include any additional metadata from the original query item
                    for key, value in query_item.items():
                        if key not in ["query", "context"]:
                            error_dict[key] = value
                            
                    technique_responses.append(error_dict)
            
            all_responses[technique] = technique_responses
            print(f"  ✓ {technique}: {len(technique_responses)} responses generated")
        
        # Save responses
        self.responses.update(all_responses)
        self._save_responses(all_responses)
        
        return all_responses
    
    def evaluate_responses(self, techniques: List[str] = None) -> Dict[str, List[Dict]]:
        """Evaluate generated responses using LLM-as-a-judge."""
        if techniques is None:
            techniques = list(self.responses.keys())
        
        print(f"Evaluating responses for {len(techniques)} techniques...")
        
        all_evaluations = {}
        
        for technique in techniques:
            if technique not in self.responses or technique not in self.judge_prompts:
                print(f"  Warning: Missing responses or judge prompt for {technique}, skipping")
                continue
            
            print(f"  Evaluating {technique} responses...")
            
            responses = self.responses[technique]
            judge_prompt_template = self.judge_prompts[technique]["prompt"]
            
            technique_evaluations = []
            
            for i, response_item in enumerate(responses):
                if "error" in response_item:
                    continue
                
                query = response_item["query"]
                response = response_item["response"]
                
                # Create evaluation prompt
                eval_prompt = f"{judge_prompt_template}\n\nQuery: {query}\nResponse to Evaluate: {response}\n\nProvide your evaluation:"
                
                try:
                    # Get evaluation from LLM judge
                    api_response = self.judge_generator.client.chat.completions.create(
                        model=self.judge_generator.model,
                        messages=[
                            {"role": "system", "content": "You are an expert evaluator of medical assistant responses."},
                            {"role": "user", "content": eval_prompt}
                        ],
                        temperature=self.judge_generator.temperature
                    )
                    evaluation = api_response.choices[0].message.content
                    
                    evaluation_dict = {
                        "query": query,
                        "response": response,
                        "evaluation": evaluation,
                        "judge_prompt_used": judge_prompt_template,
                        "technique": technique,
                        "timestamp": datetime.now().isoformat(),
                        "query_index": i
                    }
                    
                    technique_evaluations.append(evaluation_dict)
                    print(f"    Evaluation {i+1}/{len(responses)} completed")
                    
                except Exception as e:
                    print(f"    Error evaluating response {i+1}: {e}")
            
            all_evaluations[technique] = technique_evaluations
            print(f"  ✓ {technique}: {len(technique_evaluations)} evaluations completed")
        
        # Save evaluations
        self.evaluations.update(all_evaluations)
        self._save_evaluations(all_evaluations)
        
        return all_evaluations
    
    def run_complete_pipeline(self, dataset_path: str, context: PromptContext, 
                             techniques: List[str] = None, query_column: str = "query", 
                             context_column: str = "context") -> Dict[str, Any]:
        """Run the complete end-to-end pipeline."""
        print("="*80)
        print("RUNNING COMPLETE PROMPT ENGINEERING PIPELINE")
        print("="*80)
        
        # Step 1: Load dataset
        print("\n1. Loading dataset...")
        queries = self.load_dataset(dataset_path, query_column, context_column)
        
        # Step 2: Generate enhanced prompts
        print("\n2. Generating enhanced prompts...")
        enhanced_prompts = self.generate_enhanced_prompts(context, techniques)
        
        # Step 3: Generate judge prompts
        print("\n3. Generating judge prompts...")
        judge_prompts = self.generate_judge_prompts(context, techniques)
        
        # Step 4: Run prompts on dataset
        print("\n4. Running prompts on dataset...")
        responses = self.run_prompts_on_dataset(queries, techniques)
        
        # Step 5: Evaluate responses
        print("\n5. Evaluating responses...")
        evaluations = self.evaluate_responses(techniques)
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETE")
        print("="*80)
        print(f"✓ Processed {len(queries)} queries")
        print(f"✓ Tested {len(enhanced_prompts)} techniques")
        print(f"✓ Generated {sum(len(r) for r in responses.values())} responses")
        print(f"✓ Completed {sum(len(e) for e in evaluations.values())} evaluations")
        print(f"✓ All outputs saved to: {self.output_dir}")
        
        return {
            "queries": queries,
            "enhanced_prompts": enhanced_prompts,
            "judge_prompts": judge_prompts,
            "responses": responses,
            "evaluations": evaluations,
            "output_directory": self.output_dir
        }
    
    def _save_prompts(self, prompts: Dict[str, Any]):
        """Save generated prompts to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.output_dir, "prompts", f"prompts_{timestamp}.json")
        with open(filepath, 'w') as f:
            json.dump(prompts, f, indent=2)
        print(f"    Saved prompts to: {filepath}")
    
    def _save_judge_prompts(self, judge_prompts: Dict[str, Any]):
        """Save judge prompts to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.output_dir, "judge_prompts", f"judge_prompts_{timestamp}.json")
        with open(filepath, 'w') as f:
            json.dump(judge_prompts, f, indent=2)
        print(f"    Saved judge prompts to: {filepath}")
    
    def _save_responses(self, responses: Dict[str, List[Dict]]):
        """Save responses to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.output_dir, "responses", f"responses_{timestamp}.json")
        with open(filepath, 'w') as f:
            json.dump(responses, f, indent=2)
        print(f"    Saved responses to: {filepath}")
    
    def _save_evaluations(self, evaluations: Dict[str, List[Dict]]):
        """Save evaluations to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.output_dir, "evaluations", f"evaluations_{timestamp}.json")
        with open(filepath, 'w') as f:
            json.dump(evaluations, f, indent=2)
        print(f"    Saved evaluations to: {filepath}") 