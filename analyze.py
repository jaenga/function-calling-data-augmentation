"""
Analysis module: Generate analysis report on augmentation and validation results
Includes:
- Function distribution analysis
- Validation pass rate
- Disagreement types (1st vs 2nd stage rejection)
- Ambiguous ratio
"""

import os
import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List
from collections import Counter

from config import (
    VALIDATED_SINGLE_PATH, VALIDATED_MULTI_PATH,
    REJECTED_SINGLE_PATH, REJECTED_MULTI_PATH,
    GENERATED_SINGLE_PATH, GENERATED_MULTI_PATH,
    DATA_DIR, ANALYSIS_LOG_PATH, OUTPUT_DIR, TOOLS
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{DATA_DIR}/analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AnalysisEngine:
    """Analyze augmentation and validation results"""
    
    def __init__(self):
        self.validated_data = None
        self.rejected_data = None
        self.generated_data = None
        self.report = {}
    
    def load_data(self) -> None:
        """Load validated, rejected, and generated data"""
        datasets = {
            "validated": [],
            "rejected": [],
            "generated": []
        }
        
        # Load validated data
        for path in [VALIDATED_SINGLE_PATH, VALIDATED_MULTI_PATH]:
            if Path(path).exists():
                df = pd.read_csv(path)
                datasets["validated"].append(df)
                logger.info(f"Loaded {len(df)} validated samples from {path}")
        
        # Load rejected data
        for path in [REJECTED_SINGLE_PATH, REJECTED_MULTI_PATH]:
            if Path(path).exists():
                df = pd.read_csv(path)
                datasets["rejected"].append(df)
                logger.info(f"Loaded {len(df)} rejected samples from {path}")
        
        # Load generated data
        for path in [GENERATED_SINGLE_PATH, GENERATED_MULTI_PATH]:
            if Path(path).exists():
                df = pd.read_csv(path)
                datasets["generated"].append(df)
                logger.info(f"Loaded {len(df)} generated samples from {path}")
        
        # Combine datasets
        self.validated_data = pd.concat(datasets["validated"], ignore_index=True) if datasets["validated"] else pd.DataFrame()
        self.rejected_data = pd.concat(datasets["rejected"], ignore_index=True) if datasets["rejected"] else pd.DataFrame()
        self.generated_data = pd.concat(datasets["generated"], ignore_index=True) if datasets["generated"] else pd.DataFrame()
        
        logger.info(f"\nTotal validated: {len(self.validated_data)}")
        logger.info(f"Total rejected: {len(self.rejected_data)}")
        logger.info(f"Total generated: {len(self.generated_data)}")
    
    def analyze_function_distribution(self) -> Dict:
        """Analyze function distribution across validated and rejected data"""
        distribution = {}
        
        # Validated distribution
        if len(self.validated_data) > 0:
            validated_counter = Counter()
            for _, row in self.validated_data.iterrows():
                for func_name, _ in self._iter_row_calls(row):
                    validated_counter[func_name] += 1
            validated_func_counts = dict(validated_counter)
            distribution["validated"] = validated_func_counts
        else:
            distribution["validated"] = {}
        
        # Rejected distribution
        if len(self.rejected_data) > 0:
            rejected_counter = Counter()
            for _, row in self.rejected_data.iterrows():
                for func_name, _ in self._iter_row_calls(row):
                    rejected_counter[func_name] += 1
            rejected_func_counts = dict(rejected_counter)
            distribution["rejected"] = rejected_func_counts
        else:
            distribution["rejected"] = {}
        
        return distribution
    
    def analyze_argument_distribution(self) -> Dict:
        """Analyze argument value distribution"""
        argument_dist = {}
        
        if len(self.validated_data) > 0:
            for _, row in self.validated_data.iterrows():
                for func_name, args in self._iter_row_calls(row):
                    if func_name not in argument_dist:
                        argument_dist[func_name] = {}

                    for key, value in args.items():
                        if key not in argument_dist[func_name]:
                            argument_dist[func_name][key] = {}

                        if value not in argument_dist[func_name][key]:
                            argument_dist[func_name][key][value] = 0

                        argument_dist[func_name][key][value] += 1
        
        return argument_dist

    def _iter_row_calls(self, row: pd.Series):
        """Yield (function_name, required_arguments) for single or multi rows."""
        if "function_calls" in row and pd.notna(row.get("function_calls")):
            try:
                calls = (
                    json.loads(row["function_calls"])
                    if isinstance(row["function_calls"], str)
                    else row["function_calls"]
                )
            except Exception:
                calls = []
            for call in calls if isinstance(calls, list) else []:
                func_name = call.get("name") or call.get("function_name")
                args = call.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                yield func_name, self._required_arguments_only(func_name, args or {})
            return

        func_name = row.get("function_name")
        if not func_name or pd.isna(func_name):
            return

        try:
            args = json.loads(row.get("arguments", "{}")) if isinstance(row.get("arguments", "{}"), str) else row.get("arguments", {})
        except Exception:
            args = {}
        yield func_name, self._required_arguments_only(func_name, args or {})

    def _format_row_label(self, row: pd.Series) -> str:
        labels = [
            self._format_call_label(func_name, args)
            for func_name, args in self._iter_row_calls(row)
        ]
        if "function_calls" in row and pd.notna(row.get("function_calls")):
            labels = sorted(labels)
        return " + ".join(labels) if labels else "unknown"

    def _format_pred_label(self, validator_pred: Dict) -> str:
        if "function_calls" in validator_pred:
            labels = []
            for call in validator_pred.get("function_calls", []):
                func_name = call.get("function_name") or call.get("name")
                args = call.get("arguments", {})
                labels.append(self._format_call_label(
                    func_name,
                    self._required_arguments_only(func_name, args or {})
                ))
            return " + ".join(sorted(labels)) if labels else "unknown"

        predicted_func = validator_pred.get("function_name")
        predicted_args = validator_pred.get("arguments", {})
        predicted_args = self._required_arguments_only(predicted_func, predicted_args)
        return self._format_call_label(predicted_func, predicted_args)
    
    def analyze_validation_pass_rate(self) -> Dict:
        """Calculate validation pass rate"""
        total = len(self.validated_data) + len(self.rejected_data)
        
        if total == 0:
            return {
                "total": 0,
                "validated": 0,
                "rejected": 0,
                "pass_rate": 0.0
            }
        
        pass_rate = (len(self.validated_data) / total) * 100
        
        return {
            "total": total,
            "validated": len(self.validated_data),
            "rejected": len(self.rejected_data),
            "pass_rate": pass_rate
        }
    
    def analyze_disagreement_types(self) -> Dict:
        """Analyze disagreement types (1st vs 2nd stage rejection)"""
        disagreement = {
            "1st_stage": 0,
            "2nd_stage": 0,
            "unknown": 0
        }
        
        if len(self.rejected_data) > 0:
            for stage in self.rejected_data['reject_stage']:
                if stage == "1st_validation":
                    disagreement["1st_stage"] += 1
                elif stage == "2nd_validation":
                    disagreement["2nd_stage"] += 1
                else:
                    disagreement["unknown"] += 1
        
        return disagreement
    
    def analyze_ambiguous_ratio(self) -> Dict:
        """Analyze ambiguous sample ratio"""
        if len(self.validated_data) == 0:
            return {
                "total": 0,
                "ambiguous": 0,
                "clear": 0,
                "ambiguous_ratio": 0.0
            }
        
        ambiguous_mask = self.validated_data['ambiguous'].notna() & (self.validated_data['ambiguous'] != "")
        ambiguous_count = ambiguous_mask.sum()
        clear_count = len(self.validated_data) - ambiguous_count
        ambiguous_ratio = (ambiguous_count / len(self.validated_data)) * 100
        
        return {
            "total": len(self.validated_data),
            "ambiguous": ambiguous_count,
            "clear": clear_count,
            "ambiguous_ratio": ambiguous_ratio
        }
    
    def analyze_confusion_matrix(self) -> Dict:
        """Analyze confusion between similar function-argument pairs"""
        confusion_matrix = {}
        
        if len(self.rejected_data) > 0:
            for _, row in self.rejected_data.iterrows():
                try:
                    validator_pred = json.loads(row['validator_pred']) if isinstance(row['validator_pred'], str) else row['validator_pred']
                except:
                    validator_pred = {}
                
                ground_truth_label = self._format_row_label(row)
                predicted_label = self._format_pred_label(validator_pred)
                
                if predicted_label != "unknown" and predicted_label != ground_truth_label:
                    key = f"{ground_truth_label} → {predicted_label}"
                    confusion_matrix[key] = confusion_matrix.get(key, 0) + 1
        
        return confusion_matrix

    @staticmethod
    def _format_call_label(function_name: str, arguments: Dict) -> str:
        """Format a function call as a stable label for confusion analysis."""
        if not function_name:
            return "unknown"
        if not arguments:
            return function_name
        args_text = ",".join(f"{key}={arguments[key]}" for key in sorted(arguments.keys()))
        return f"{function_name}({args_text})"

    @staticmethod
    def _required_arguments_only(function_name: str, arguments: Dict) -> Dict:
        """Keep only required parameters so optional args do not appear as confusion."""
        if not function_name or function_name not in TOOLS:
            return arguments or {}
        required = TOOLS[function_name].get("parameters", {}).get("required", [])
        return {key: arguments[key] for key in required if key in arguments}
    
    def generate_report(self) -> Dict:
        """Generate complete analysis report"""
        self.load_data()
        
        self.report = {
            "summary": {
                "validation": self.analyze_validation_pass_rate(),
                "ambiguous": self.analyze_ambiguous_ratio(),
                "disagreement": self.analyze_disagreement_types()
            },
            "distribution": {
                "function": self.analyze_function_distribution(),
                "argument": self.analyze_argument_distribution()
            },
            "confusion": self.analyze_confusion_matrix()
        }
        
        return self.report
    
    def print_report(self) -> None:
        """Print formatted analysis report"""
        if not self.report:
            return
        
        print("\n" + "="*80)
        print("DATA AUGMENTATION & VALIDATION ANALYSIS REPORT")
        print("="*80)
        
        # Summary
        print("\n[SUMMARY]")
        validation = self.report["summary"]["validation"]
        print(f"Total Samples: {validation['total']}")
        print(f"  ✓ Validated: {validation['validated']} ({validation['pass_rate']:.1f}%)")
        print(f"  ✗ Rejected:  {validation['rejected']} ({100-validation['pass_rate']:.1f}%)")
        
        # Ambiguous
        ambiguous = self.report["summary"]["ambiguous"]
        print(f"\nAmbiguous Samples: {ambiguous['ambiguous']} / {ambiguous['total']} ({ambiguous['ambiguous_ratio']:.1f}%)")
        print(f"  Clear: {ambiguous['clear']}")
        
        # Disagreement
        disagreement = self.report["summary"]["disagreement"]
        total_rejected = disagreement["1st_stage"] + disagreement["2nd_stage"] + disagreement["unknown"]
        if total_rejected > 0:
            print(f"\nRejection Stages:")
            print(f"  1st Stage: {disagreement['1st_stage']} ({100*disagreement['1st_stage']/total_rejected:.1f}%)")
            print(f"  2nd Stage: {disagreement['2nd_stage']} ({100*disagreement['2nd_stage']/total_rejected:.1f}%)")
            if disagreement["unknown"] > 0:
                print(f"  Unknown:   {disagreement['unknown']}")
        
        # Function Distribution
        print("\n[FUNCTION DISTRIBUTION]")
        func_dist = self.report["distribution"]["function"]
        
        if func_dist["validated"]:
            print("\nValidated by Function:")
            for func_name in sorted(func_dist["validated"].keys()):
                count = func_dist["validated"][func_name]
                print(f"  {func_name:40s}: {count:3d}")
        
        if func_dist["rejected"]:
            print("\nRejected by Function:")
            for func_name in sorted(func_dist["rejected"].keys()):
                count = func_dist["rejected"][func_name]
                print(f"  {func_name:40s}: {count:3d}")
        
        # Argument Distribution
        print("\n[ARGUMENT DISTRIBUTION]")
        arg_dist = self.report["distribution"]["argument"]
        
        for func_name in sorted(arg_dist.keys()):
            print(f"\n{func_name}:")
            for arg_name in sorted(arg_dist[func_name].keys()):
                print(f"  {arg_name}:")
                for arg_value in sorted(arg_dist[func_name][arg_name].keys()):
                    count = arg_dist[func_name][arg_name][arg_value]
                    print(f"    {arg_value:30s}: {count:3d}")
        
        # Confusion Matrix
        confusion = self.report["confusion"]
        if confusion:
            print("\n[CONFUSION MATRIX]")
            print("Function Prediction Errors:")
            for confusion_pair in sorted(confusion.keys()):
                count = confusion[confusion_pair]
                print(f"  {confusion_pair:60s}: {count:3d}")
        
        print("\n" + "="*80 + "\n")
    
    def save_report(self, output_path: str = ANALYSIS_LOG_PATH) -> None:
        """Save report to file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("DATA AUGMENTATION & VALIDATION ANALYSIS REPORT\n")
            f.write("="*80 + "\n")
            
            # Summary
            f.write("\n[SUMMARY]\n")
            validation = self.report["summary"]["validation"]
            f.write(f"Total Samples: {validation['total']}\n")
            f.write(f"  ✓ Validated: {validation['validated']} ({validation['pass_rate']:.1f}%)\n")
            f.write(f"  ✗ Rejected:  {validation['rejected']} ({100-validation['pass_rate']:.1f}%)\n")
            
            # Ambiguous
            ambiguous = self.report["summary"]["ambiguous"]
            f.write(f"\nAmbiguous Samples: {ambiguous['ambiguous']} / {ambiguous['total']} ({ambiguous['ambiguous_ratio']:.1f}%)\n")
            f.write(f"  Clear: {ambiguous['clear']}\n")
            
            # Disagreement
            disagreement = self.report["summary"]["disagreement"]
            total_rejected = disagreement["1st_stage"] + disagreement["2nd_stage"] + disagreement["unknown"]
            if total_rejected > 0:
                f.write(f"\nRejection Stages:\n")
                f.write(f"  1st Stage: {disagreement['1st_stage']} ({100*disagreement['1st_stage']/total_rejected:.1f}%)\n")
                f.write(f"  2nd Stage: {disagreement['2nd_stage']} ({100*disagreement['2nd_stage']/total_rejected:.1f}%)\n")
                if disagreement["unknown"] > 0:
                    f.write(f"  Unknown:   {disagreement['unknown']}\n")
            
            # Function Distribution
            f.write("\n[FUNCTION DISTRIBUTION]\n")
            func_dist = self.report["distribution"]["function"]
            
            if func_dist["validated"]:
                f.write("\nValidated by Function:\n")
                for func_name in sorted(func_dist["validated"].keys()):
                    count = func_dist["validated"][func_name]
                    f.write(f"  {func_name:40s}: {count:3d}\n")
            
            if func_dist["rejected"]:
                f.write("\nRejected by Function:\n")
                for func_name in sorted(func_dist["rejected"].keys()):
                    count = func_dist["rejected"][func_name]
                    f.write(f"  {func_name:40s}: {count:3d}\n")
            
            # Argument Distribution
            f.write("\n[ARGUMENT DISTRIBUTION]\n")
            arg_dist = self.report["distribution"]["argument"]
            
            for func_name in sorted(arg_dist.keys()):
                f.write(f"\n{func_name}:\n")
                for arg_name in sorted(arg_dist[func_name].keys()):
                    f.write(f"  {arg_name}:\n")
                    for arg_value in sorted(arg_dist[func_name][arg_name].keys()):
                        count = arg_dist[func_name][arg_name][arg_value]
                        f.write(f"    {arg_value:30s}: {count:3d}\n")
            
            # Confusion Matrix
            confusion = self.report["confusion"]
            if confusion:
                f.write("\n[CONFUSION MATRIX]\n")
                f.write("Function Prediction Errors:\n")
                for confusion_pair in sorted(confusion.keys()):
                    count = confusion[confusion_pair]
                    f.write(f"  {confusion_pair:60s}: {count:3d}\n")
            
            f.write("\n" + "="*80 + "\n")
        
        logger.info(f"✅ Report saved to {output_path}")


def main():
    """Main analysis pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze augmentation and validation results")
    parser.add_argument("--save", action="store_true", help="Save report to file")
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting Analysis Pipeline")
    
    engine = AnalysisEngine()
    engine.generate_report()
    engine.print_report()
    
    if args.save:
        engine.save_report()
    
    logger.info("✨ Analysis completed!")


if __name__ == "__main__":
    main()
