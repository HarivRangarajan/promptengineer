"""
Learnable Demonstration Generator with Human-in-the-Loop Feedback

This module provides the DemonstrationGenerator class that automatically generates,
refines, and improves demonstrations to clarify task instructions for LLMs.
It supports error feedback integration and human-in-the-loop refinement.
"""

import json
import os
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import re
from openai import OpenAI


@dataclass
class Demonstration:
    """Represents a single demonstration example."""
    id: str
    content: str
    type: str  # "positive" or "negative"
    error_category: Optional[str] = None
    status: str = "pending"  # "pending", "accepted", "rejected", "edited"
    feedback: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    edited_content: Optional[str] = None
    is_seed: bool = False  # Whether this is a seed demonstration


@dataclass
class SeedDemonstration:
    """Represents a seed demonstration that guides generation structure."""
    content: str
    type: str  # "positive" or "negative"
    description: Optional[str] = None  # Human description of what makes this a good example
    structure_notes: Optional[str] = None  # Notes about the structure to replicate
    
    def to_demonstration(self, task_id: str, demo_id: str) -> Demonstration:
        """Convert to a regular Demonstration object."""
        return Demonstration(
            id=demo_id,
            content=self.content,
            type=self.type,
            status="accepted",  # Seed demonstrations are pre-accepted
            is_seed=True
        )


@dataclass
class ErrorCategory:
    """Represents an error category from feedback."""
    name: str
    description: str
    examples: List[str] = field(default_factory=list)
    priority: int = 1  # Higher number = higher priority
    count: int = 0


@dataclass
class DemonstrationSet:
    """A set of demonstrations for a specific task or error category."""
    task_description: str
    positive_demonstrations: List[Demonstration] = field(default_factory=list)
    negative_demonstrations: List[Demonstration] = field(default_factory=list)
    error_categories: List[ErrorCategory] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    seed_demonstrations: List[SeedDemonstration] = field(default_factory=list)  # Added seed demonstrations


class DemonstrationGenerator:
    """
    Main class for generating, refining, and managing demonstrations with
    human-in-the-loop feedback and error category integration.
    """
    
    def __init__(self, 
                 api_key: str, 
                 model: str = "gpt-4o",
                 output_dir: str = "demonstration_output",
                 num_positive_demos: int = 2,
                 num_negative_demos: int = 2):
        """
        Initialize the DemonstrationGenerator.
        
        Args:
            api_key: OpenAI API key
            model: LLM model to use for generation
            output_dir: Directory to save demonstrations and audit trails
            num_positive_demos: Default number of positive demonstrations
            num_negative_demos: Default number of negative demonstrations
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.output_dir = Path(output_dir)
        self.num_positive_demos = num_positive_demos
        self.num_negative_demos = num_negative_demos
        
        # Create output directory structure
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "demonstrations").mkdir(exist_ok=True)
        (self.output_dir / "audit_trails").mkdir(exist_ok=True)
        (self.output_dir / "feedback").mkdir(exist_ok=True)
        
        # Internal state
        self.demonstration_sets: Dict[str, DemonstrationSet] = {}
        self.audit_trail: List[Dict[str, Any]] = []
        
    def load_prompt_file(self, prompt_file_path: str) -> Dict[str, str]:
        """
        Load and parse a prompt file containing task instructions.
        
        Args:
            prompt_file_path: Path to the prompt file (.txt or .md)
            
        Returns:
            Dictionary with parsed prompt components
        """
        try:
            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try to parse structured content
            sections = self._parse_prompt_sections(content)
            
            return {
                "raw_content": content,
                "task_description": sections.get("task_description", ""),
                "instructions": sections.get("instructions", ""),
                "constraints": sections.get("constraints", ""),
                "context": sections.get("context", ""),
                "file_path": prompt_file_path
            }
        except Exception as e:
            raise ValueError(f"Failed to load prompt file {prompt_file_path}: {str(e)}")
    
    def _parse_prompt_sections(self, content: str) -> Dict[str, str]:
        """Parse prompt content into structured sections."""
        sections = {}
        
        # Common section patterns
        patterns = {
            "task_description": r"(?i)(?:task|description|objective):\s*(.*?)(?=\n\n|\n[A-Z]|$)",
            "instructions": r"(?i)instructions?:\s*(.*?)(?=\n\n|\n[A-Z]|$)",
            "constraints": r"(?i)constraints?:\s*(.*?)(?=\n\n|\n[A-Z]|$)",
            "context": r"(?i)context:\s*(.*?)(?=\n\n|\n[A-Z]|$)"
        }
        
        for section, pattern in patterns.items():
            match = re.search(pattern, content, re.DOTALL)
            if match:
                sections[section] = match.group(1).strip()
        
        # If no structured sections found, use the entire content as task description
        if not sections:
            sections["task_description"] = content.strip()
        
        return sections
    
    def load_error_feedback(self, feedback_file_path: str) -> List[ErrorCategory]:
        """
        Load and parse error feedback file containing error categories.
        
        Args:
            feedback_file_path: Path to feedback file (.txt, .md, or .json)
            
        Returns:
            List of ErrorCategory objects
        """
        try:
            with open(feedback_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if feedback_file_path.endswith('.json'):
                return self._parse_json_feedback(content)
            else:
                return self._parse_text_feedback(content)
                
        except Exception as e:
            raise ValueError(f"Failed to load feedback file {feedback_file_path}: {str(e)}")
    
    def _parse_json_feedback(self, content: str) -> List[ErrorCategory]:
        """Parse JSON format feedback (from LLM-as-a-judge output)."""
        try:
            data = json.loads(content)
            error_categories = []
            
            # Handle different JSON structures
            if isinstance(data, list):
                # List of evaluation results
                category_counts = {}
                for item in data:
                    category = item.get('category', 'unknown')
                    reason = item.get('reason', '')
                    if category != 'none' and item.get('label') == 0:  # Failed cases
                        if category not in category_counts:
                            category_counts[category] = {'count': 0, 'examples': []}
                        category_counts[category]['count'] += 1
                        category_counts[category]['examples'].append(reason)
                
                # Convert to ErrorCategory objects
                for category, data in category_counts.items():
                    error_categories.append(ErrorCategory(
                        name=category.upper(),
                        description=self._get_category_description(category),
                        examples=data['examples'][:3],  # Top 3 examples
                        count=data['count'],
                        priority=self._get_category_priority(category)
                    ))
            
            elif isinstance(data, dict):
                # Summary report format
                if 'technique_analysis' in data:
                    # Extract error patterns from technique analysis
                    for technique, analysis in data['technique_analysis'].items():
                        if 'failure_categories' in analysis:
                            for category, count in analysis['failure_categories'].items():
                                error_categories.append(ErrorCategory(
                                    name=category.upper(),
                                    description=self._get_category_description(category),
                                    count=count,
                                    priority=self._get_category_priority(category)
                                ))
            
            return error_categories
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in feedback file: {str(e)}")
    
    def _parse_text_feedback(self, content: str) -> List[ErrorCategory]:
        """Parse text/markdown format feedback."""
        error_categories = []
        
        # Look for category patterns
        category_pattern = r'(?i)(?:category|error)\s*([a-z])?\s*[-:]?\s*(.*?)(?=\n(?:category|error|\n)|\Z)'
        matches = re.findall(category_pattern, content, re.DOTALL)
        
        for i, (category_letter, description) in enumerate(matches):
            if not category_letter:
                category_letter = chr(ord('A') + i)  # Assign letters if missing
            
            # Extract examples from description
            examples = []
            example_pattern = r'(?i)(?:example|fail if|fail when):\s*(.*?)(?=\n|$)'
            example_matches = re.findall(example_pattern, description)
            examples.extend([ex.strip() for ex in example_matches if ex.strip()])
            
            error_categories.append(ErrorCategory(
                name=f"CATEGORY_{category_letter.upper()}",
                description=description.strip(),
                examples=examples,
                priority=self._get_category_priority(category_letter)
            ))
        
        return error_categories
    
    def _get_category_description(self, category: str) -> str:
        """Get standard descriptions for known error categories."""
        descriptions = {
            'a': 'No answer/irrelevant to the question',
            'b': 'Incorrect or contradictory information',
            'd': 'Visual assumptions without visual access',
            'e': 'Inappropriate "I don\'t know" responses',
            'f': 'Premature advancement to future steps'
        }
        return descriptions.get(category.lower(), f"Error category {category}")
    
    def _get_category_priority(self, category: str) -> int:
        """Get priority levels for error categories (higher = more important)."""
        priorities = {
            'd': 5,  # Visual assumptions - highest priority
            'f': 4,  # Premature advancement
            'b': 3,  # Incorrect information
            'e': 2,  # Inappropriate "I don't know"
            'a': 1   # Irrelevant - lowest priority
        }
        return priorities.get(category.lower(), 1)
    
    def load_seed_demonstrations(self, seed_data: Union[str, List[Dict[str, Any]], List[SeedDemonstration]]) -> List[SeedDemonstration]:
        """
        Load seed demonstrations from various sources.
        
        Args:
            seed_data: Can be:
                - String path to JSON file containing seed demonstrations
                - List of dictionaries with seed demonstration data
                - List of SeedDemonstration objects
                
        Returns:
            List of SeedDemonstration objects
        """
        if isinstance(seed_data, str):
            # Load from file
            try:
                with open(seed_data, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return self._parse_seed_demonstrations_from_dict(data)
            except Exception as e:
                raise ValueError(f"Failed to load seed demonstrations from {seed_data}: {str(e)}")
        
        elif isinstance(seed_data, list):
            if not seed_data:
                return []
            
            # Check if it's already SeedDemonstration objects
            if isinstance(seed_data[0], SeedDemonstration):
                return seed_data
            
            # Assume it's a list of dictionaries
            return self._parse_seed_demonstrations_from_dict(seed_data)
        
        else:
            raise ValueError("seed_data must be a file path, list of dictionaries, or list of SeedDemonstration objects")
    
    def _parse_seed_demonstrations_from_dict(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> List[SeedDemonstration]:
        """Parse seed demonstrations from dictionary format."""
        seed_demonstrations = []
        
        # Handle different data structures
        if isinstance(data, dict):
            # Check for common structures
            if 'seed_demonstrations' in data:
                demonstrations_data = data['seed_demonstrations']
            elif 'demonstrations' in data:
                demonstrations_data = data['demonstrations']
            elif 'positive_examples' in data or 'negative_examples' in data:
                demonstrations_data = []
                if 'positive_examples' in data:
                    for example in data['positive_examples']:
                        demonstrations_data.append({
                            'content': example,
                            'type': 'positive'
                        })
                if 'negative_examples' in data:
                    for example in data['negative_examples']:
                        demonstrations_data.append({
                            'content': example,
                            'type': 'negative'
                        })
            else:
                # Assume the entire dict is a single demonstration
                demonstrations_data = [data]
        else:
            # List of demonstrations
            demonstrations_data = data
        
        # Convert to SeedDemonstration objects
        for demo_data in demonstrations_data:
            if isinstance(demo_data, str):
                # Just content, assume positive
                seed_demonstrations.append(SeedDemonstration(
                    content=demo_data,
                    type='positive'
                ))
            elif isinstance(demo_data, dict):
                seed_demonstrations.append(SeedDemonstration(
                    content=demo_data.get('content', ''),
                    type=demo_data.get('type', 'positive'),
                    description=demo_data.get('description'),
                    structure_notes=demo_data.get('structure_notes')
                ))
        
        return seed_demonstrations
    
    def create_seed_demonstration(self, 
                                content: str, 
                                demo_type: str = 'positive', 
                                description: Optional[str] = None,
                                structure_notes: Optional[str] = None) -> SeedDemonstration:
        """
        Create a single seed demonstration programmatically.
        
        Args:
            content: The demonstration content
            demo_type: "positive" or "negative"
            description: Optional description of what makes this a good example
            structure_notes: Optional notes about structure to replicate
            
        Returns:
            SeedDemonstration object
        """
        return SeedDemonstration(
            content=content,
            type=demo_type,
            description=description,
            structure_notes=structure_notes
        )
    
    def generate_initial_demonstrations(self, 
                                      prompt_data: Dict[str, str],
                                      task_id: Optional[str] = None,
                                      seed_demonstrations: Optional[List[SeedDemonstration]] = None) -> DemonstrationSet:
        """
        Generate initial set of demonstrations for a given prompt, optionally using seed demonstrations.
        
        Args:
            prompt_data: Parsed prompt data from load_prompt_file
            task_id: Optional identifier for this task
            seed_demonstrations: Optional seed demonstrations to guide generation structure
            
        Returns:
            DemonstrationSet with generated demonstrations
        """
        if not task_id:
            task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Process seed demonstrations if provided
        if seed_demonstrations is None:
            seed_demonstrations = []
        
        # Create meta-prompt for demonstration generation (now includes seeds)
        meta_prompt = self._create_demonstration_meta_prompt(prompt_data, seed_demonstrations)
        
        # Start with seed demonstrations converted to regular demonstrations
        positive_demos = []
        negative_demos = []
        
        # Convert seed demonstrations to regular demonstrations
        for i, seed_demo in enumerate(seed_demonstrations):
            demo_id = f"{task_id}_seed_{seed_demo.type}_{i+1}"
            regular_demo = seed_demo.to_demonstration(task_id, demo_id)
            
            if seed_demo.type == "positive":
                positive_demos.append(regular_demo)
            else:
                negative_demos.append(regular_demo)
        
        # Calculate how many additional demonstrations to generate
        additional_positive = max(0, self.num_positive_demos - len([d for d in seed_demonstrations if d.type == "positive"]))
        additional_negative = max(0, self.num_negative_demos - len([d for d in seed_demonstrations if d.type == "negative"]))
        
        # Generate additional positive demonstrations if needed
        if additional_positive > 0:
            generated_positive = self._generate_demonstrations(
                meta_prompt, 
                "positive", 
                additional_positive,
                task_id,
                seed_demonstrations
            )
            positive_demos.extend(generated_positive)
        
        # Generate additional negative demonstrations if needed
        if additional_negative > 0:
            generated_negative = self._generate_demonstrations(
                meta_prompt, 
                "negative", 
                additional_negative,
                task_id,
                seed_demonstrations
            )
            negative_demos.extend(generated_negative)
        
        # Create demonstration set
        demo_set = DemonstrationSet(
            task_description=prompt_data.get("task_description", ""),
            positive_demonstrations=positive_demos,
            negative_demonstrations=negative_demos,
            seed_demonstrations=seed_demonstrations,
            metadata={
                "prompt_file": prompt_data.get("file_path", ""),
                "created_at": datetime.now().isoformat(),
                "generation_method": "seed_guided_generation" if seed_demonstrations else "initial_llm_generation",
                "seed_demonstrations_count": len(seed_demonstrations),
                "generated_demonstrations_count": additional_positive + additional_negative
            }
        )
        
        # Store and save
        self.demonstration_sets[task_id] = demo_set
        self._save_demonstration_set(task_id, demo_set)
        self._log_audit_event("initial_generation", task_id, {
            "positive_count": len(positive_demos),
            "negative_count": len(negative_demos),
            "seed_count": len(seed_demonstrations),
            "generated_count": additional_positive + additional_negative
        })
        
        return demo_set
    
    def generate_demonstrations_with_seeds(self,
                                         prompt_data: Dict[str, str],
                                         seed_demonstrations: Union[str, List[Dict[str, Any]], List[SeedDemonstration]],
                                         task_id: Optional[str] = None) -> DemonstrationSet:
        """
        Convenience method to generate demonstrations using seed demonstrations.
        
        Args:
            prompt_data: Parsed prompt data from load_prompt_file
            seed_demonstrations: Seed demonstrations in any supported format
            task_id: Optional identifier for this task
            
        Returns:
            DemonstrationSet with demonstrations guided by seeds
        """
        # Load and parse seed demonstrations
        parsed_seeds = self.load_seed_demonstrations(seed_demonstrations)
        
        # Generate demonstrations using the seeds
        return self.generate_initial_demonstrations(
            prompt_data=prompt_data,
            task_id=task_id,
            seed_demonstrations=parsed_seeds
        )
    
    def add_seed_to_existing_task(self, 
                                task_id: str, 
                                seed_demonstrations: Union[str, List[Dict[str, Any]], List[SeedDemonstration]]) -> DemonstrationSet:
        """
        Add seed demonstrations to an existing task and regenerate demonstrations.
        
        Args:
            task_id: Existing task identifier
            seed_demonstrations: Seed demonstrations to add
            
        Returns:
            Updated DemonstrationSet
        """
        if task_id not in self.demonstration_sets:
            raise ValueError(f"Task {task_id} not found")
        
        # Load and parse seed demonstrations
        parsed_seeds = self.load_seed_demonstrations(seed_demonstrations)
        
        # Get existing demo set
        demo_set = self.demonstration_sets[task_id]
        
        # Add seeds to the demonstration set
        demo_set.seed_demonstrations.extend(parsed_seeds)
        
        # Update metadata
        demo_set.metadata["seed_demonstrations_added_at"] = datetime.now().isoformat()
        demo_set.metadata["total_seed_demonstrations"] = len(demo_set.seed_demonstrations)
        
        # Save updated demonstration set
        self._save_demonstration_set(task_id, demo_set)
        self._log_audit_event("seed_demonstrations_added", task_id, {
            "new_seeds_count": len(parsed_seeds),
            "total_seeds": len(demo_set.seed_demonstrations)
        })
        
        return demo_set
    
    def get_seed_demonstrations_summary(self, task_id: str) -> Dict[str, Any]:
        """
        Get a summary of seed demonstrations for a task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Dictionary with seed demonstration summary
        """
        if task_id not in self.demonstration_sets:
            raise ValueError(f"Task {task_id} not found")
        
        demo_set = self.demonstration_sets[task_id]
        seeds = demo_set.seed_demonstrations
        
        positive_seeds = [s for s in seeds if s.type == "positive"]
        negative_seeds = [s for s in seeds if s.type == "negative"]
        
        return {
            "task_id": task_id,
            "total_seeds": len(seeds),
            "positive_seeds": len(positive_seeds),
            "negative_seeds": len(negative_seeds),
            "seeds_with_descriptions": len([s for s in seeds if s.description]),
            "seeds_with_structure_notes": len([s for s in seeds if s.structure_notes]),
            "seed_contents_preview": [
                {
                    "type": seed.type,
                    "content_preview": seed.content[:100] + "..." if len(seed.content) > 100 else seed.content,
                    "has_description": bool(seed.description),
                    "has_structure_notes": bool(seed.structure_notes)
                }
                for seed in seeds
            ]
        }
    
    def _create_demonstration_meta_prompt(self, prompt_data: Dict[str, str], seed_demonstrations: List[SeedDemonstration]) -> str:
        """Create meta-prompt for demonstration generation."""
        seed_examples_str = ""
        if seed_demonstrations:
            seed_examples_str = "Here are some seed demonstrations to guide the generation:\n\n"
            for i, seed_demo in enumerate(seed_demonstrations):
                seed_examples_str += f"Seed {i+1} ({seed_demo.type.upper()}):\n"
                seed_examples_str += f"Content: {seed_demo.content}\n"
                if seed_demo.description:
                    seed_examples_str += f"Description: {seed_demo.description}\n"
                if seed_demo.structure_notes:
                    seed_examples_str += f"Structure Notes: {seed_demo.structure_notes}\n"
                seed_examples_str += "-------------------------\n"
        
        return f"""You are an expert at creating clear, instructive demonstrations for LLM training.

TASK CONTEXT:
{prompt_data.get('task_description', 'No description provided')}

INSTRUCTIONS:
{prompt_data.get('instructions', 'No specific instructions provided')}

CONSTRAINTS:
{prompt_data.get('constraints', 'No specific constraints provided')}

{seed_examples_str}

Your job is to create demonstration examples that help another LLM understand how to correctly follow these instructions.

For positive demonstrations: Show clear, correct examples of following the instructions properly.
For negative demonstrations: Show common mistakes or misunderstandings that violate the instructions.

Each demonstration should:
1. Be realistic and practical
2. Clearly illustrate the key principles
3. Be appropriately detailed but concise
4. Show the reasoning behind correct/incorrect approaches

Focus on the most important aspects that an LLM might misunderstand or execute incorrectly."""
    
    def _generate_demonstrations(self, 
                               meta_prompt: str, 
                               demo_type: str, 
                               count: int,
                               task_id: str,
                               seed_demonstrations: Optional[List[SeedDemonstration]] = None) -> List[Demonstration]:
        """Generate demonstrations using LLM."""
        demonstrations = []
        
        # Get relevant seed demonstrations for this type
        relevant_seeds = []
        if seed_demonstrations:
            relevant_seeds = [seed for seed in seed_demonstrations if seed.type == demo_type]
        
        for i in range(count):
            # Create specific prompt for this demonstration
            specific_prompt = f"""{meta_prompt}

Generate a {"positive (correct)" if demo_type == "positive" else "negative (incorrect)"} demonstration example.

This should be demonstration #{i+1} of {count} {demo_type} examples.
Make it distinct from the others while illustrating the same core principles."""

            # Add guidance from seed demonstrations if available
            if relevant_seeds:
                specific_prompt += f"""

IMPORTANT: Use the seed demonstrations above as structural and stylistic guidance.
Follow similar patterns in terms of:
- Content structure and organization
- Level of detail and explanation
- Tone and language style
- Format and presentation

The seed demonstrations show the preferred style - generate new content that follows these patterns but addresses different scenarios or aspects of the task."""

            specific_prompt += """

Return only the demonstration content, clearly structured and ready to use as training data."""

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert demonstration generator for LLM training."},
                        {"role": "user", "content": specific_prompt}
                    ],
                    temperature=0.7 if demo_type == "negative" else 0.3,  # More variation for negative examples
                    max_tokens=1000
                )
                
                content = response.choices[0].message.content.strip()
                
                demo = Demonstration(
                    id=f"{task_id}_{demo_type}_{i+1}",
                    content=content,
                    type=demo_type
                )
                
                demonstrations.append(demo)
                
            except Exception as e:
                print(f"Warning: Failed to generate {demo_type} demonstration {i+1}: {str(e)}")
        
        return demonstrations
    
    def refine_demonstrations_with_feedback(self, 
                                          task_id: str, 
                                          error_categories: List[ErrorCategory]) -> DemonstrationSet:
        """
        Refine demonstrations based on error feedback categories.
        
        Args:
            task_id: Identifier for the task/demonstration set
            error_categories: List of error categories to address
            
        Returns:
            Updated DemonstrationSet with refined demonstrations
        """
        if task_id not in self.demonstration_sets:
            raise ValueError(f"Task {task_id} not found")
        
        demo_set = self.demonstration_sets[task_id]
        demo_set.error_categories = error_categories
        
        # Sort error categories by priority (highest first)
        error_categories.sort(key=lambda x: x.priority, reverse=True)
        
        # Generate contrastive demonstrations for each error category
        for error_category in error_categories:
            if error_category.count > 0:  # Only address categories with actual errors
                self._generate_contrastive_demonstrations(demo_set, error_category, task_id)
        
        # Save updated demonstration set
        self._save_demonstration_set(task_id, demo_set)
        self._log_audit_event("refinement_with_feedback", task_id, {
            "error_categories_addressed": len(error_categories),
            "total_error_count": sum(cat.count for cat in error_categories)
        })
        
        return demo_set
    
    def _generate_contrastive_demonstrations(self, 
                                           demo_set: DemonstrationSet, 
                                           error_category: ErrorCategory,
                                           task_id: str):
        """Generate contrastive demonstration pairs for a specific error category."""
        # Create targeted meta-prompt for this error category
        meta_prompt = f"""You are creating targeted demonstration examples to address a specific error pattern.

TASK: {demo_set.task_description}

ERROR CATEGORY: {error_category.name}
DESCRIPTION: {error_category.description}
ERROR COUNT: {error_category.count} instances

EXAMPLE ERROR PATTERNS:
{chr(10).join(f"- {example}" for example in error_category.examples[:3])}

Create contrastive demonstration pairs that specifically address this error pattern:

1. POSITIVE EXAMPLE: Show the correct way to handle situations that typically trigger this error
2. NEGATIVE EXAMPLE: Show exactly what NOT to do (the error pattern itself)

Each example should:
- Directly address the error pattern
- Be clear about why it's correct/incorrect
- Help an LLM avoid this specific mistake
- Include brief explanation of the reasoning"""

        # Generate positive demonstration
        positive_prompt = f"""{meta_prompt}

Generate the POSITIVE (correct) demonstration that shows how to avoid the "{error_category.name}" error.
Focus on the correct approach that prevents this specific error pattern."""

        # Generate negative demonstration
        negative_prompt = f"""{meta_prompt}

Generate the NEGATIVE (incorrect) demonstration that shows the "{error_category.name}" error pattern.
This should illustrate exactly what the LLM did wrong in the failed cases."""

        try:
            # Generate positive example
            pos_response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at creating targeted training demonstrations."},
                    {"role": "user", "content": positive_prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            
            pos_demo = Demonstration(
                id=f"{task_id}_refined_positive_{error_category.name.lower()}",
                content=pos_response.choices[0].message.content.strip(),
                type="positive",
                error_category=error_category.name
            )
            
            # Generate negative example
            neg_response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at creating targeted training demonstrations."},
                    {"role": "user", "content": negative_prompt}
                ],
                temperature=0.5,
                max_tokens=800
            )
            
            neg_demo = Demonstration(
                id=f"{task_id}_refined_negative_{error_category.name.lower()}",
                content=neg_response.choices[0].message.content.strip(),
                type="negative",
                error_category=error_category.name
            )
            
            # Add to demonstration set
            demo_set.positive_demonstrations.append(pos_demo)
            demo_set.negative_demonstrations.append(neg_demo)
            
        except Exception as e:
            print(f"Warning: Failed to generate contrastive demonstrations for {error_category.name}: {str(e)}")
    
    def present_for_human_review(self, task_id: str) -> Dict[str, Any]:
        """
        Present demonstrations to user for review and feedback.
        
        Args:
            task_id: Identifier for the task/demonstration set
            
        Returns:
            Dictionary with review interface data
        """
        if task_id not in self.demonstration_sets:
            raise ValueError(f"Task {task_id} not found")
        
        demo_set = self.demonstration_sets[task_id]
        
        review_data = {
            "task_id": task_id,
            "task_description": demo_set.task_description,
            "demonstrations": [],
            "error_categories": [
                {
                    "name": cat.name,
                    "description": cat.description,
                    "count": cat.count,
                    "priority": cat.priority
                }
                for cat in demo_set.error_categories
            ],
            "review_instructions": self._get_review_instructions()
        }
        
        # Prepare demonstrations for review
        all_demos = demo_set.positive_demonstrations + demo_set.negative_demonstrations
        for demo in all_demos:
            review_data["demonstrations"].append({
                "id": demo.id,
                "content": demo.content,
                "type": demo.type,
                "error_category": demo.error_category,
                "status": demo.status,
                "feedback": demo.feedback
            })
        
        return review_data
    
    def _get_review_instructions(self) -> str:
        """Get instructions for human reviewers."""
        return """
DEMONSTRATION REVIEW INSTRUCTIONS:

For each demonstration, please:
1. ACCEPT: If the demonstration clearly illustrates the intended concept
2. REJECT: If the demonstration is unclear, incorrect, or unhelpful
3. EDIT: If the demonstration has potential but needs modifications

When editing:
- Preserve the core teaching intent
- Improve clarity and accuracy
- Ensure it addresses the specific error category (if applicable)

When providing feedback:
- Be specific about what needs improvement
- Suggest concrete changes
- Explain why changes are needed

Your feedback helps improve the demonstration quality for future LLM training.
"""
    
    def process_human_feedback(self, 
                             task_id: str, 
                             feedback_data: List[Dict[str, Any]]) -> DemonstrationSet:
        """
        Process human feedback on demonstrations.
        
        Args:
            task_id: Identifier for the task/demonstration set
            feedback_data: List of feedback items with demo_id, action, and optional edits/feedback
            
        Returns:
            Updated DemonstrationSet
        """
        if task_id not in self.demonstration_sets:
            raise ValueError(f"Task {task_id} not found")
        
        demo_set = self.demonstration_sets[task_id]
        
        # Create lookup for demonstrations
        all_demos = demo_set.positive_demonstrations + demo_set.negative_demonstrations
        demo_lookup = {demo.id: demo for demo in all_demos}
        
        processed_feedback = []
        
        for feedback_item in feedback_data:
            demo_id = feedback_item.get("demo_id")
            action = feedback_item.get("action")  # "accept", "reject", "edit"
            user_feedback = feedback_item.get("feedback", "")
            edited_content = feedback_item.get("edited_content", "")
            
            if demo_id not in demo_lookup:
                continue
            
            demo = demo_lookup[demo_id]
            demo.status = action
            demo.feedback = user_feedback
            
            if action == "edit" and edited_content:
                demo.edited_content = edited_content
            
            processed_feedback.append({
                "demo_id": demo_id,
                "action": action,
                "original_content": demo.content[:100] + "..." if len(demo.content) > 100 else demo.content,
                "feedback": user_feedback,
                "has_edits": bool(edited_content)
            })
        
        # Save updated demonstration set
        self._save_demonstration_set(task_id, demo_set)
        self._log_audit_event("human_feedback_processed", task_id, {
            "feedback_items": len(processed_feedback),
            "accepted": len([f for f in processed_feedback if f["action"] == "accept"]),
            "rejected": len([f for f in processed_feedback if f["action"] == "reject"]),
            "edited": len([f for f in processed_feedback if f["action"] == "edit"])
        })
        
        return demo_set
    
    def get_final_demonstrations(self, task_id: str) -> Dict[str, Any]:
        """
        Get the final set of demonstrations ready for use in prompt engineering.
        
        Args:
            task_id: Identifier for the task/demonstration set
            
        Returns:
            Dictionary with final demonstrations and metadata
        """
        if task_id not in self.demonstration_sets:
            raise ValueError(f"Task {task_id} not found")
        
        demo_set = self.demonstration_sets[task_id]
        
        # Filter accepted and edited demonstrations
        accepted_positive = []
        accepted_negative = []
        
        for demo in demo_set.positive_demonstrations:
            if demo.status == "accepted" or (demo.status == "edit" and demo.edited_content):
                content = demo.edited_content if demo.edited_content else demo.content
                accepted_positive.append({
                    "id": demo.id,
                    "content": content,
                    "error_category": demo.error_category,
                    "type": "positive",
                    "is_seed": demo.is_seed
                })
        
        for demo in demo_set.negative_demonstrations:
            if demo.status == "accepted" or (demo.status == "edit" and demo.edited_content):
                content = demo.edited_content if demo.edited_content else demo.content
                accepted_negative.append({
                    "id": demo.id,
                    "content": content,
                    "error_category": demo.error_category,
                    "type": "negative",
                    "is_seed": demo.is_seed
                })
        
        final_output = {
            "task_id": task_id,
            "task_description": demo_set.task_description,
            "positive_demonstrations": accepted_positive,
            "negative_demonstrations": accepted_negative,
            "total_demonstrations": len(accepted_positive) + len(accepted_negative),
            "error_categories_addressed": [cat.name for cat in demo_set.error_categories],
            "generation_metadata": demo_set.metadata,
            "ready_for_prompt_integration": True,
            "formatted_for_llm": self._format_demonstrations_for_llm(accepted_positive, accepted_negative)
        }
        
        # Save final output
        final_file = self.output_dir / f"final_demonstrations_{task_id}.json"
        with open(final_file, 'w') as f:
            json.dump(final_output, f, indent=2)
        
        return final_output
    
    def _format_demonstrations_for_llm(self, positive_demos: List[Dict], negative_demos: List[Dict]) -> str:
        """Format demonstrations for inclusion in LLM prompts."""
        formatted = "DEMONSTRATION EXAMPLES:\n\n"
        
        if positive_demos:
            formatted += "✅ CORRECT EXAMPLES (Do this):\n"
            for i, demo in enumerate(positive_demos, 1):
                seed_indicator = " [SEED]" if demo.get('is_seed', False) else ""
                formatted += f"\n{i}. {demo['content']}{seed_indicator}\n"
        
        if negative_demos:
            formatted += "\n❌ INCORRECT EXAMPLES (Avoid this):\n"
            for i, demo in enumerate(negative_demos, 1):
                seed_indicator = " [SEED]" if demo.get('is_seed', False) else ""
                formatted += f"\n{i}. {demo['content']}{seed_indicator}\n"
        
        formatted += "\nFollow the patterns shown in the correct examples and avoid the mistakes shown in the incorrect examples.\n"
        
        return formatted
    
    def _save_demonstration_set(self, task_id: str, demo_set: DemonstrationSet):
        """Save demonstration set to file."""
        output_file = self.output_dir / "demonstrations" / f"{task_id}.json"
        
        # Convert to serializable format
        data = {
            "task_description": demo_set.task_description,
            "positive_demonstrations": [
                {
                    "id": demo.id,
                    "content": demo.content,
                    "type": demo.type,
                    "error_category": demo.error_category,
                    "status": demo.status,
                    "feedback": demo.feedback,
                    "created_at": demo.created_at,
                    "edited_content": demo.edited_content
                }
                for demo in demo_set.positive_demonstrations
            ],
            "negative_demonstrations": [
                {
                    "id": demo.id,
                    "content": demo.content,
                    "type": demo.type,
                    "error_category": demo.error_category,
                    "status": demo.status,
                    "feedback": demo.feedback,
                    "created_at": demo.created_at,
                    "edited_content": demo.edited_content
                }
                for demo in demo_set.negative_demonstrations
            ],
            "error_categories": [
                {
                    "name": cat.name,
                    "description": cat.description,
                    "examples": cat.examples,
                    "priority": cat.priority,
                    "count": cat.count
                }
                for cat in demo_set.error_categories
            ],
            "metadata": demo_set.metadata,
            "seed_demonstrations": [
                {
                    "content": sd.content,
                    "type": sd.type,
                    "description": sd.description,
                    "structure_notes": sd.structure_notes
                } for sd in demo_set.seed_demonstrations
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _log_audit_event(self, event_type: str, task_id: str, details: Dict[str, Any]):
        """Log audit events for traceability."""
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "task_id": task_id,
            "details": details
        }
        
        self.audit_trail.append(audit_entry)
        
        # Save to audit file
        audit_file = self.output_dir / "audit_trails" / f"audit_{task_id}.json"
        with open(audit_file, 'w') as f:
            json.dump(self.audit_trail, f, indent=2)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about demonstration generation and feedback."""
        stats = {
            "total_tasks": len(self.demonstration_sets),
            "total_demonstrations": 0,
            "total_seed_demonstrations": 0,
            "demonstration_status_counts": {"pending": 0, "accepted": 0, "rejected": 0, "edited": 0},
            "seed_demonstration_counts": {"positive": 0, "negative": 0},
            "error_categories_addressed": set(),
            "audit_events": len(self.audit_trail)
        }
        
        for demo_set in self.demonstration_sets.values():
            all_demos = demo_set.positive_demonstrations + demo_set.negative_demonstrations
            stats["total_demonstrations"] += len(all_demos)
            
            # Count seed demonstrations
            stats["total_seed_demonstrations"] += len(demo_set.seed_demonstrations)
            for seed in demo_set.seed_demonstrations:
                stats["seed_demonstration_counts"][seed.type] += 1
            
            for demo in all_demos:
                stats["demonstration_status_counts"][demo.status] += 1
            
            for cat in demo_set.error_categories:
                stats["error_categories_addressed"].add(cat.name)
        
        stats["error_categories_addressed"] = list(stats["error_categories_addressed"])
        
        return stats 