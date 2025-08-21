"""
Feedback Integrator for mohs-llm-as-a-judge Error Categories

This module provides specialized integration with the mohs-llm-as-a-judge
error categorization system to seamlessly extract and process error feedback
for demonstration generation.
"""

import json
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass


@dataclass
class JudgeErrorAnalysis:
    """Analysis results from LLM-as-a-judge evaluation."""
    technique_name: str
    total_responses: int
    pass_rate: float
    failure_categories: Dict[str, int]
    common_failure_patterns: List[str]
    priority_errors: List[str]


class FeedbackIntegrator:
    """
    Specialized integrator for mohs-llm-as-a-judge error feedback.
    
    This class knows how to parse and extract error categories from
    the specific output formats used by the mohs-llm-as-a-judge system.
    """
    
    def __init__(self):
        """Initialize the feedback integrator."""
        self.category_mappings = {
            'a': {
                'name': 'IRRELEVANT_RESPONSE',
                'description': 'Response completely ignores the question and provides unrelated information',
                'priority': 1
            },
            'b': {
                'name': 'INCORRECT_INFORMATION',
                'description': 'Response contains fabricated, incorrect, or contradictory information',
                'priority': 3
            },
            'd': {
                'name': 'VISUAL_ASSUMPTIONS',
                'description': 'Response assumes visual knowledge that LLM should not have',
                'priority': 5
            },
            'e': {
                'name': 'INAPPROPRIATE_UNCERTAINTY',
                'description': 'LLM claims "I don\'t know" when answer should be available',
                'priority': 2
            },
            'f': {
                'name': 'PREMATURE_ADVANCEMENT',
                'description': 'LLM provides instructions for future steps prematurely',
                'priority': 4
            }
        }
    
    def extract_from_judge_output(self, judge_output_path: str) -> List[JudgeErrorAnalysis]:
        """
        Extract error analysis from mohs-llm-as-a-judge output files.
        
        Args:
            judge_output_path: Path to judge output directory or file
            
        Returns:
            List of JudgeErrorAnalysis objects
        """
        output_path = Path(judge_output_path)
        
        if output_path.is_file():
            return self._process_single_file(output_path)
        elif output_path.is_dir():
            return self._process_directory(output_path)
        else:
            raise ValueError(f"Invalid path: {judge_output_path}")
    
    def _process_single_file(self, file_path: Path) -> List[JudgeErrorAnalysis]:
        """Process a single judge output file."""
        if file_path.suffix == '.json':
            return self._process_json_file(file_path)
        elif file_path.suffix == '.csv':
            return self._process_csv_file(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def _process_directory(self, dir_path: Path) -> List[JudgeErrorAnalysis]:
        """Process all judge output files in a directory."""
        analyses = []
        
        # Look for summary reports first
        summary_files = list(dir_path.glob("*summary_report*.json"))
        if summary_files:
            for summary_file in summary_files:
                analyses.extend(self._process_summary_report(summary_file))
        
        # Look for detailed evaluation files
        eval_files = list(dir_path.glob("*evaluation*.csv")) + list(dir_path.glob("*results*.csv"))
        for eval_file in eval_files:
            analyses.extend(self._process_csv_file(eval_file))
        
        return analyses
    
    def _process_json_file(self, file_path: Path) -> List[JudgeErrorAnalysis]:
        """Process JSON file from judge output."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if 'technique_analysis' in data:
            return self._process_summary_report(file_path)
        else:
            # Individual evaluation results
            return self._process_evaluation_results(data)
    
    def _process_summary_report(self, file_path: Path) -> List[JudgeErrorAnalysis]:
        """Process summary report JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        analyses = []
        
        if 'technique_analysis' in data:
            for technique, analysis in data['technique_analysis'].items():
                # Extract failure categories
                failure_categories = {}
                common_patterns = []
                
                if 'failure_categories' in analysis:
                    failure_categories = analysis['failure_categories']
                
                # Extract detailed failure info if available
                if 'detailed_failures' in analysis:
                    for failure in analysis['detailed_failures'][:5]:  # Top 5 patterns
                        if 'reason' in failure:
                            common_patterns.append(failure['reason'])
                
                # Determine priority errors based on frequency and severity
                priority_errors = self._identify_priority_errors(failure_categories)
                
                analyses.append(JudgeErrorAnalysis(
                    technique_name=technique.replace('technique_', '').replace('_', ' ').title(),
                    total_responses=analysis.get('total_responses', 0),
                    pass_rate=analysis.get('pass_rate', 0.0),
                    failure_categories=failure_categories,
                    common_failure_patterns=common_patterns,
                    priority_errors=priority_errors
                ))
        
        return analyses
    
    def _process_csv_file(self, file_path: Path) -> List[JudgeErrorAnalysis]:
        """Process CSV file with evaluation results."""
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Warning: Could not read CSV file {file_path}: {e}")
            return []
        
        analyses = []
        
        # Group by technique if available
        if 'technique' in df.columns or any('technique' in col.lower() for col in df.columns):
            technique_col = next((col for col in df.columns if 'technique' in col.lower()), 'technique')
            
            for technique in df[technique_col].unique():
                if pd.isna(technique):
                    continue
                    
                technique_data = df[df[technique_col] == technique]
                analysis = self._analyze_technique_data(technique_data, str(technique))
                if analysis:
                    analyses.append(analysis)
        else:
            # Single technique or no technique grouping
            analysis = self._analyze_technique_data(df, "Unknown Technique")
            if analysis:
                analyses.append(analysis)
        
        return analyses
    
    def _analyze_technique_data(self, df: pd.DataFrame, technique_name: str) -> Optional[JudgeErrorAnalysis]:
        """Analyze data for a specific technique."""
        if df.empty:
            return None
        
        # Find label and category columns
        label_col = self._find_column(df, ['label', 'llm_judge_label', 'pass_fail', 'result'])
        category_col = self._find_column(df, ['category', 'failure_category', 'primary_failure', 'error_category'])
        reason_col = self._find_column(df, ['reason', 'explanation', 'failure_reason'])
        
        if not label_col:
            return None
        
        total_responses = len(df)
        
        # Calculate pass rate
        if df[label_col].dtype == 'object':
            # Handle string labels
            passed = len(df[df[label_col].str.lower().isin(['pass', '1', 'true', 'yes'])])
        else:
            # Handle numeric labels
            passed = len(df[df[label_col] == 1])
        
        pass_rate = passed / total_responses if total_responses > 0 else 0.0
        
        # Analyze failure categories
        failure_categories = {}
        common_patterns = []
        
        if category_col:
            failed_data = df[df[label_col] != 1] if df[label_col].dtype != 'object' else df[~df[label_col].str.lower().isin(['pass', '1', 'true', 'yes'])]
            
            if not failed_data.empty:
                category_counts = failed_data[category_col].value_counts().to_dict()
                failure_categories = {str(k): int(v) for k, v in category_counts.items() if pd.notna(k)}
        
        # Extract common failure patterns
        if reason_col:
            failed_data = df[df[label_col] != 1] if df[label_col].dtype != 'object' else df[~df[label_col].str.lower().isin(['pass', '1', 'true', 'yes'])]
            
            if not failed_data.empty:
                reasons = failed_data[reason_col].dropna().tolist()
                common_patterns = reasons[:5]  # Top 5 most recent patterns
        
        priority_errors = self._identify_priority_errors(failure_categories)
        
        return JudgeErrorAnalysis(
            technique_name=technique_name,
            total_responses=total_responses,
            pass_rate=pass_rate,
            failure_categories=failure_categories,
            common_failure_patterns=common_patterns,
            priority_errors=priority_errors
        )
    
    def _find_column(self, df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
        """Find column by possible names (case insensitive)."""
        df_columns_lower = [col.lower() for col in df.columns]
        
        for name in possible_names:
            if name.lower() in df_columns_lower:
                return df.columns[df_columns_lower.index(name.lower())]
            
            # Partial match
            for col in df.columns:
                if name.lower() in col.lower():
                    return col
        
        return None
    
    def _process_evaluation_results(self, data: List[Dict]) -> List[JudgeErrorAnalysis]:
        """Process individual evaluation results."""
        if not isinstance(data, list):
            data = [data]
        
        # Group by technique if available
        technique_groups = {}
        
        for item in data:
            technique = item.get('technique', 'Unknown')
            if technique not in technique_groups:
                technique_groups[technique] = []
            technique_groups[technique].append(item)
        
        analyses = []
        for technique, items in technique_groups.items():
            analysis = self._analyze_evaluation_items(items, technique)
            if analysis:
                analyses.append(analysis)
        
        return analyses
    
    def _analyze_evaluation_items(self, items: List[Dict], technique_name: str) -> Optional[JudgeErrorAnalysis]:
        """Analyze individual evaluation items."""
        if not items:
            return None
        
        total_responses = len(items)
        passed = sum(1 for item in items if item.get('label') == 1)
        pass_rate = passed / total_responses
        
        # Count failure categories
        failure_categories = {}
        common_patterns = []
        
        for item in items:
            if item.get('label') == 0:  # Failed case
                category = item.get('category', 'unknown')
                reason = item.get('reason', '')
                
                if category and category != 'none':
                    failure_categories[category] = failure_categories.get(category, 0) + 1
                
                if reason:
                    common_patterns.append(reason)
        
        priority_errors = self._identify_priority_errors(failure_categories)
        
        return JudgeErrorAnalysis(
            technique_name=technique_name,
            total_responses=total_responses,
            pass_rate=pass_rate,
            failure_categories=failure_categories,
            common_failure_patterns=common_patterns[:5],
            priority_errors=priority_errors
        )
    
    def _identify_priority_errors(self, failure_categories: Dict[str, int]) -> List[str]:
        """Identify priority errors based on frequency and severity."""
        if not failure_categories:
            return []
        
        # Sort by priority (using predefined priorities) and frequency
        prioritized = []
        
        for category, count in failure_categories.items():
            category_lower = category.lower()
            priority = 1  # Default priority
            
            # Map to known priorities
            for cat_key, cat_info in self.category_mappings.items():
                if cat_key in category_lower or cat_info['name'].lower() in category_lower:
                    priority = cat_info['priority']
                    break
            
            prioritized.append((category, count, priority))
        
        # Sort by priority (descending) then by count (descending)
        prioritized.sort(key=lambda x: (x[2], x[1]), reverse=True)
        
        # Return top priority error categories
        return [item[0] for item in prioritized[:3]]
    
    def convert_to_demonstration_feedback(self, analyses: List[JudgeErrorAnalysis]) -> Dict[str, Any]:
        """
        Convert judge error analysis to format suitable for demonstration generator.
        
        Args:
            analyses: List of JudgeErrorAnalysis objects
            
        Returns:
            Dictionary with error feedback formatted for demonstration generator
        """
        feedback = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "source": "mohs-llm-as-a-judge",
            "techniques_analyzed": [],
            "aggregated_errors": {},
            "priority_categories": [],
            "recommendations": []
        }
        
        all_failures = {}
        all_patterns = []
        
        for analysis in analyses:
            # Add technique summary
            feedback["techniques_analyzed"].append({
                "technique": analysis.technique_name,
                "total_responses": analysis.total_responses,
                "pass_rate": analysis.pass_rate,
                "failure_categories": analysis.failure_categories,
                "priority_errors": analysis.priority_errors
            })
            
            # Aggregate failures across techniques
            for category, count in analysis.failure_categories.items():
                if category not in all_failures:
                    all_failures[category] = {"total_count": 0, "techniques": {}}
                all_failures[category]["total_count"] += count
                all_failures[category]["techniques"][analysis.technique_name] = count
            
            # Collect patterns
            all_patterns.extend(analysis.common_failure_patterns)
        
        feedback["aggregated_errors"] = all_failures
        
        # Identify overall priority categories
        priority_categories = []
        for category, data in all_failures.items():
            category_info = self._get_category_info(category)
            priority_categories.append({
                "category": category,
                "name": category_info["name"],
                "description": category_info["description"],
                "total_count": data["total_count"],
                "priority": category_info["priority"],
                "affected_techniques": list(data["techniques"].keys())
            })
        
        # Sort by priority and count
        priority_categories.sort(key=lambda x: (x["priority"], x["total_count"]), reverse=True)
        feedback["priority_categories"] = priority_categories
        
        # Generate recommendations
        feedback["recommendations"] = self._generate_recommendations(priority_categories, analyses)
        
        return feedback
    
    def _get_category_info(self, category: str) -> Dict[str, Any]:
        """Get standardized information for an error category."""
        category_lower = category.lower()
        
        # Try to match with known categories
        for cat_key, cat_info in self.category_mappings.items():
            if cat_key in category_lower or category_lower in cat_info["name"].lower():
                return cat_info
        
        # Default for unknown categories
        return {
            "name": category.upper(),
            "description": f"Error category: {category}",
            "priority": 1
        }
    
    def _generate_recommendations(self, priority_categories: List[Dict], analyses: List[JudgeErrorAnalysis]) -> List[str]:
        """Generate recommendations based on error analysis."""
        recommendations = []
        
        if not priority_categories:
            return ["No significant error patterns detected."]
        
        top_category = priority_categories[0]
        recommendations.append(
            f"Focus on addressing '{top_category['name']}' errors, which occurred {top_category['total_count']} times "
            f"across {len(top_category['affected_techniques'])} techniques."
        )
        
        if len(priority_categories) > 1:
            second_category = priority_categories[1]
            recommendations.append(
                f"Secondary focus: '{second_category['name']}' errors ({second_category['total_count']} occurrences)."
            )
        
        # Technique-specific recommendations
        worst_technique = min(analyses, key=lambda x: x.pass_rate)
        if worst_technique.pass_rate < 0.7:
            recommendations.append(
                f"The '{worst_technique.technique_name}' technique shows the lowest pass rate ({worst_technique.pass_rate:.1%}) "
                f"and may benefit from additional demonstrations."
            )
        
        # Pattern-based recommendations
        if priority_categories[0]["category"].lower() in ["d", "visual"]:
            recommendations.append(
                "Consider adding demonstrations that explicitly show how to handle visual-related queries "
                "without assuming visual capabilities."
            )
        elif priority_categories[0]["category"].lower() in ["b", "incorrect"]:
            recommendations.append(
                "Focus on demonstrations that emphasize accuracy and fact-checking, "
                "particularly around medical/safety information."
            )
        
        return recommendations 