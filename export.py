"""
Export module: Convert validated data to JSONL format for fine-tuning
Supports Qwen format (FunctionGemma format is TODO)
Includes train/valid/test split (80/10/10)
"""

import os
import json
import logging
import uuid
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split

from config import (
    TOOLS, SPLIT_RATIOS,
    VALIDATED_SINGLE_PATH, VALIDATED_MULTI_PATH,
    TRAIN_JSONL_PATH, VALID_JSONL_PATH, TEST_JSONL_PATH,
    DATA_DIR, OUTPUT_DIR
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{DATA_DIR}/export.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class QwenFormatter:
    """Convert data to Qwen function calling format"""
    
    @staticmethod
    def build_tools_schema() -> List[Dict]:
        """Build tools schema for Qwen format"""
        tools = []
        
        for func_name, func_def in TOOLS.items():
            tool = {
                "type": "function",
                "function": {
                    "name": func_name,
                    "description": func_def["description"],
                    "parameters": func_def["parameters"]
                }
            }
            tools.append(tool)
        
        return tools
    
    @staticmethod
    def format_sample(utterance: str, function_name: str, arguments: Dict) -> Dict:
        """
        Convert a sample to Qwen format
        
        Qwen format structure:
        {
            "messages": [
                {"role": "user", "content": "user_utterance"},
                {"role": "assistant", "content": null, "tool_calls": [{"id": "...", "type": "function", "function": {"name": "...", "arguments": "..."}}]}
            ],
            "tools": [tool_definitions]
        }
        """
        # Parse arguments if needed
        if isinstance(arguments, str):
            try:
                args_dict = json.loads(arguments)
            except:
                args_dict = {}
        else:
            args_dict = arguments
        
        # Build tool call
        tool_call = {
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": function_name,
                "arguments": json.dumps(args_dict, ensure_ascii=False)
            }
        }
        
        # Build messages
        messages = [
            {
                "role": "user",
                "content": utterance
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [tool_call]
            }
        ]
        
        # Build complete sample
        sample = {
            "messages": messages,
            "tools": QwenFormatter.build_tools_schema()
        }
        
        return sample

    @staticmethod
    def format_multi_sample(utterance: str, function_calls: str | List[Dict]) -> Dict:
        """Convert a multi-function sample to Qwen format with multiple tool calls."""
        if isinstance(function_calls, str):
            try:
                calls = json.loads(function_calls)
            except Exception:
                calls = []
        else:
            calls = function_calls or []

        tool_calls = []
        for call in calls:
            func_name = call.get("name") or call.get("function_name")
            arguments = call.get("arguments", {})
            if not func_name:
                continue
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except Exception:
                    arguments = {}
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": json.dumps(arguments or {}, ensure_ascii=False)
                }
            })

        messages = [
            {
                "role": "user",
                "content": utterance
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls
            }
        ]

        return {
            "messages": messages,
            "tools": QwenFormatter.build_tools_schema()
        }


class FunctionGemmaFormatter:
    """Convert data to FunctionGemma format (TODO)"""
    
    @staticmethod
    def format_sample(utterance: str, function_name: str, arguments: Dict) -> Dict:
        """
        TODO: Convert a sample to FunctionGemma format
        
        Placeholder - implement when FunctionGemma format is finalized
        """
        # TODO: FunctionGemma format implementation
        # FunctionGemma uses a different schema, likely:
        # {
        #     "prompt": "user_utterance",
        #     "completion": "{\"type\": \"function\", \"function\": {...}}"
        # }
        raise NotImplementedError("FunctionGemma format not yet implemented")


class ExportPipeline:
    """Main export pipeline"""
    
    def __init__(self):
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        self.train_samples = []
        self.valid_samples = []
        self.test_samples = []
    
    def load_validated_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load validated data from both single and multi"""
        datasets = []
        
        for path in [VALIDATED_SINGLE_PATH, VALIDATED_MULTI_PATH]:
            if Path(path).exists():
                if Path(path).stat().st_size <= 1:
                    logger.warning(f"Skipping empty validated file: {path}")
                    continue
                df = pd.read_csv(path)
                if df.empty:
                    logger.warning(f"Skipping validated file with no rows: {path}")
                    continue
                logger.info(f"Loaded {len(df)} samples from {path}")
                datasets.append(df)
            else:
                logger.warning(f"File not found: {path}")
        
        if not datasets:
            logger.error("No validated data found!")
            return pd.DataFrame(), pd.DataFrame()
        
        # Combine datasets
        all_data = pd.concat(datasets, ignore_index=True)
        logger.info(f"Total validated samples: {len(all_data)}")
        
        return all_data, pd.DataFrame()
    
    def split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/valid/test
        Ratios: 80/10/10
        """
        train_ratio = SPLIT_RATIOS["train"]
        valid_ratio = SPLIT_RATIOS["valid"]
        test_ratio = SPLIT_RATIOS["test"]
        
        # First split: train and temp (valid + test)
        train_data, temp_data = train_test_split(
            data, 
            test_size=(1 - train_ratio),
            random_state=42
        )
        
        # Second split: valid and test
        # Calculate ratio for second split
        valid_test_ratio = test_ratio / (valid_ratio + test_ratio)
        valid_data, test_data = train_test_split(
            temp_data,
            test_size=valid_test_ratio,
            random_state=42
        )
        
        logger.info(f"Split data:")
        logger.info(f"  Train: {len(train_data)} ({100*len(train_data)/len(data):.1f}%)")
        logger.info(f"  Valid: {len(valid_data)} ({100*len(valid_data)/len(data):.1f}%)")
        logger.info(f"  Test:  {len(test_data)} ({100*len(test_data)/len(data):.1f}%)")
        
        return train_data, valid_data, test_data
    
    def export_qwen_jsonl(self) -> None:
        """Export data in Qwen format to JSONL"""
        logger.info("\n" + "="*80)
        logger.info("EXPORTING TO QWEN FORMAT")
        logger.info("="*80)
        
        # Load validated data
        all_data, _ = self.load_validated_data()
        
        if len(all_data) == 0:
            logger.error("No data to export!")
            return
        if len(all_data) < 3:
            logger.warning(f"Not enough data to split/export ({len(all_data)} samples). Skipping export.")
            return
        
        # Split data
        train_data, valid_data, test_data = self.split_data(all_data)
        
        # Export each split
        splits = {
            "train": (train_data, TRAIN_JSONL_PATH),
            "valid": (valid_data, VALID_JSONL_PATH),
            "test": (test_data, TEST_JSONL_PATH)
        }
        
        for split_name, (split_data, output_path) in splits.items():
            logger.info(f"\nExporting {split_name} set ({len(split_data)} samples)...")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for idx, row in split_data.iterrows():
                    if "function_calls" in row and pd.notna(row.get("function_calls")):
                        sample = QwenFormatter.format_multi_sample(
                            utterance=row['user_query'],
                            function_calls=row['function_calls']
                        )
                    else:
                        sample = QwenFormatter.format_sample(
                            utterance=row['user_query'],
                            function_name=row['function_name'],
                            arguments=row['arguments']
                        )
                    
                    # Write as JSONL
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            logger.info(f"  ✓ Saved to {output_path}")
        
        logger.info("\n✅ Qwen format export completed!")
    
    def export_functiongemma_jsonl(self) -> None:
        """Export data in FunctionGemma format to JSONL (TODO)"""
        logger.info("\n" + "="*80)
        logger.info("EXPORTING TO FUNCTIONGEMMA FORMAT (TODO)")
        logger.info("="*80)
        
        # TODO: Implement FunctionGemma export when format is finalized
        logger.warning("FunctionGemma format export is not yet implemented.")
        logger.warning("Placeholder files will be created.")
        
        # Create placeholder files
        for path in [TRAIN_JSONL_PATH, VALID_JSONL_PATH, TEST_JSONL_PATH]:
            path_gemma = path.replace(".jsonl", "_gemma.jsonl")
            with open(path_gemma, 'w', encoding='utf-8') as f:
                f.write("# TODO: FunctionGemma format implementation\n")
    
    def export(self, format: str = "qwen") -> None:
        """
        Main export pipeline
        format: "qwen" or "functiongemma" or "all"
        """
        logger.info("🚀 Starting Export Pipeline")
        
        if format in ["qwen", "all"]:
            self.export_qwen_jsonl()
        
        if format in ["functiongemma", "all"]:
            self.export_functiongemma_jsonl()
        
        logger.info("\n✨ Export pipeline completed!")


def main():
    """Main export pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Export validated mission data to JSONL")
    parser.add_argument("--format", choices=["qwen", "functiongemma", "all"], default="qwen",
                        help="Export format (default: qwen)")
    
    args = parser.parse_args()
    
    pipeline = ExportPipeline()
    pipeline.export(format=args.format)


if __name__ == "__main__":
    main()
