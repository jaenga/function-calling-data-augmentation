"""
Validation module: 2-stage validation using OpenAI function calling
1st stage: Validate function calling
2nd stage: Semantic validation with exact match requirement
"""

import os
import json
import logging
import time
import pandas as pd
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional
from openai import OpenAI

from config import (
    MODEL_NAME, OPENAI_API_KEY, TOOLS, VALIDATION_RULES,
    GENERATED_SINGLE_PATH, GENERATED_MULTI_PATH,
    VALIDATED_SINGLE_PATH, VALIDATED_MULTI_PATH,
    REJECTED_SINGLE_PATH, REJECTED_MULTI_PATH,
    API_ERROR_SINGLE_PATH, API_ERROR_MULTI_PATH,
    DATA_DIR, VALIDATION_WARNING_LOG_PATH
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{DATA_DIR}/validation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Warning log handler
warning_handler = logging.FileHandler(VALIDATION_WARNING_LOG_PATH)
warning_handler.setFormatter(logging.Formatter('%(message)s'))
warning_logger = logging.getLogger("warning")
warning_logger.addHandler(warning_handler)
warning_logger.setLevel(logging.WARNING)


class OpenAIAPIError(Exception):
    """Raised when OpenAI remains unavailable after retry attempts."""


def _tool_parameters(func_name: str) -> Dict:
    return TOOLS[func_name].get("parameters", {"type": "object", "properties": {}})


def _tool_required(func_name: str) -> List[str]:
    return _tool_parameters(func_name).get("required", [])


def _required_arguments_only(func_name: str, arguments: Dict) -> Dict:
    required = _tool_required(func_name)
    return {key: arguments[key] for key in required if key in arguments}


def _normalize_call(call: Dict) -> Dict:
    func_name = call.get("function_name") or call.get("name")
    args = call.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {}
    return {
        "function_name": func_name,
        "arguments": _required_arguments_only(func_name, args or {})
    }


def call_openai_with_retry(client, model, messages, max_retries=5, **kwargs):
    """Call OpenAI with simple backoff for rate limits and temporary overload."""
    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            time.sleep(2)
            return response
        except Exception as e:
            err = str(e)
            last_error = err
            if "429" in err or "RESOURCE_EXHAUSTED" in err or "rate limit" in err.lower():
                wait = 60 * (attempt + 1)
                logger.warning(f"Rate limit. {wait}초 대기...")
                time.sleep(wait)
            elif "503" in err or "Service Unavailable" in err or "UNAVAILABLE" in err:
                wait = 10 * (attempt + 1)
                logger.warning(f"서버 혼잡. {wait}초 대기...")
                time.sleep(wait)
            else:
                raise
    raise OpenAIAPIError(f"최대 재시도 횟수 초과: {last_error}")


class RuleBasedValidator:
    """Rule-based validation to detect warning cases"""
    
    @staticmethod
    def check_warning_rules(utterance: str, predicted_func: str, predicted_args: Dict) -> List[str]:
        """
        Check rule-based warnings
        Returns: list of warning messages (empty if no warnings)
        """
        warnings = []
        
        # Rule 1: Negative sentiment + success prediction
        negative_keywords = ["못", "안 했", "안했", "실패", "못했어", "못했습니다"]
        has_negative = any(keyword in utterance for keyword in negative_keywords)
        
        if has_negative and predicted_func == "submit_mission_result":
            predicted_result = predicted_args.get("result_type")
            if predicted_result == "success":
                warnings.append(
                    f"⚠️  NEGATIVE_WITH_SUCCESS: '{utterance}' → "
                    f"predicted success but contains negative sentiment"
                )
        
        # Rule 2: Place mention without place equivalency check
        place_keywords = ["집", "카페", "공원", "도서관", "회사", "학교", "집에서", "카페에서", "공원에서"]
        has_place_mention = any(keyword in utterance for keyword in place_keywords)
        
        if has_place_mention and predicted_func == "check_mission_equivalency":
            predicted_type = predicted_args.get("equivalency_type")
            if predicted_type != "place":
                warnings.append(
                    f"⚠️  PLACE_MENTION_WITHOUT_PLACE_CHECK: '{utterance}' → "
                    f"mentions place but predicted {predicted_type} equivalency"
                )
        
        return warnings


class OpenAIValidator1st:
    """1st stage validation: Function calling validation"""
    
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
    
    def _build_function_schema(self) -> list:
        """Build tool schema for OpenAI function calling."""
        tools = []
        
        for func_name, func_def in TOOLS.items():
            tool = {
                "type": "function",
                "function": {
                    "name": func_name,
                    "description": func_def["description"],
                    "parameters": _tool_parameters(func_name)
                }
            }
            tools.append(tool)
        
        return tools
    
    def validate(self, utterance: str) -> Tuple[Optional[Dict], bool]:
        """
        1st stage validation: Use OpenAI function calling
        Returns: (validation_result, is_valid)
        validation_result: {"function_name": str, "arguments": dict} or None
        is_valid: True if function calling succeeded
        """
        try:
            tools_schema = self._build_function_schema()
            
            response = call_openai_with_retry(
                client=self.client,
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": f"Analyze this user utterance and call the appropriate function:\n\n{utterance}"
                    }
                ],
                tools=tools_schema,
                tool_choice="required"
            )
            
            tool_calls = response.choices[0].message.tool_calls or []
            if tool_calls:
                tool_call = tool_calls[0]
                result = {
                    "function_name": tool_call.function.name,
                    "arguments": json.loads(tool_call.function.arguments or "{}")
                }
                return result, True
            
            return None, False
            
        except OpenAIAPIError:
            raise
        except Exception as e:
            logger.error(f"1st validation error: {str(e)}")
            return None, False

    def validate_multi(self, utterance: str) -> Tuple[List[Dict], bool]:
        """
        1st stage validation for multi-function utterances.
        Returns all tool calls in model-provided order.
        """
        try:
            tools_schema = self._build_function_schema()

            response = call_openai_with_retry(
                client=self.client,
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Analyze this user utterance and call every appropriate "
                            "function in order. If the utterance has two intents, "
                            "return two tool calls:\n\n"
                            f"{utterance}"
                        )
                    }
                ],
                tools=tools_schema,
                tool_choice="required"
            )

            tool_calls = response.choices[0].message.tool_calls or []
            results = []
            for tool_call in tool_calls:
                results.append({
                    "function_name": tool_call.function.name,
                    "arguments": json.loads(tool_call.function.arguments or "{}")
                })

            return results, bool(results)

        except OpenAIAPIError:
            raise
        except Exception as e:
            logger.error(f"1st multi validation error: {str(e)}")
            return [], False
    
    def exact_match(self, predicted: Dict, ground_truth: Dict) -> bool:
        """
        Exact match validation: both function_name and arguments must match
        """
        if not predicted or not ground_truth:
            return False
        
        # Check function name
        if predicted["function_name"] != ground_truth["function_name"]:
            return False
        
        # Check only required arguments. Optional production parameters are ignored.
        pred_args = predicted.get("arguments", {})
        gt_args = ground_truth.get("arguments", {})
        
        # Normalize: convert string JSON to dict if needed
        if isinstance(gt_args, str):
            try:
                gt_args = json.loads(gt_args)
            except:
                pass
        
        func_name = ground_truth["function_name"]
        pred_required = _required_arguments_only(func_name, pred_args)
        gt_required = _required_arguments_only(func_name, gt_args)
        
        required_keys = _tool_required(func_name)
        if any(key not in pred_required for key in required_keys):
            return False
        
        return pred_required == gt_required

    def exact_match_many(self, predicted_calls: List[Dict], ground_truth_calls: List[Dict]) -> bool:
        """Exact match for multi-function calls: count and required args must match, order ignored."""
        if len(predicted_calls) != len(ground_truth_calls):
            return False

        def call_key(call: Dict) -> Tuple[str, str]:
            normalized = _normalize_call(call)
            args_json = json.dumps(
                normalized["arguments"],
                ensure_ascii=False,
                sort_keys=True
            )
            return normalized["function_name"], args_json

        return Counter(call_key(call) for call in predicted_calls) == Counter(
            call_key(call) for call in ground_truth_calls
        )


class OpenAIValidator2nd:
    """2nd stage validation: Semantic validation using native function calling"""
    
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def _build_validation_schema(self) -> list:
        """Build native function calling schema for semantic validation decisions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "mark_mapping_valid",
                    "description": "Mark the utterance-to-function mapping as correct and clear.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "mark_mapping_ambiguous",
                    "description": "Mark the mapping as acceptable but ambiguous, with a concise reason.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "Concise ambiguity description."
                            }
                        },
                        "required": ["reason"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "mark_mapping_invalid",
                    "description": "Mark the mapping as incorrect, with a concise reason.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "Concise invalidity description."
                            }
                        },
                        "required": ["reason"]
                    }
                }
            }
        ]
    
    def validate(self, utterance: str, func_name: str, arguments: Dict) -> Tuple[bool, Optional[str]]:
        """
        2nd stage validation: Semantic correctness check
        Returns: (is_valid, ambiguity_description)
        """
        try:
            # Build semantic check prompt
            args_str = json.dumps(arguments, ensure_ascii=False)
            prompt = f"""Analyze if this user utterance correctly maps to the function call.

Utterance: "{utterance}"
Function: {func_name}
Arguments: {args_str}

Questions to check:
1. Does the utterance clearly indicate the intention to call this function with these specific arguments?
2. Is there any ambiguity in interpreting the utterance?
3. Could the utterance reasonably be interpreted as calling a different function or with different arguments?

Call exactly one validation decision function:
- mark_mapping_valid if the mapping is correct and clear
- mark_mapping_ambiguous if the mapping is acceptable but ambiguous
- mark_mapping_invalid if the mapping is wrong
"""
            
            tools_schema = self._build_validation_schema()
            response = call_openai_with_retry(
                client=self.client,
                model=MODEL_NAME,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                tools=tools_schema,
                tool_choice="required"
            )

            tool_calls = response.choices[0].message.tool_calls or []
            if tool_calls:
                tool_call = tool_calls[0]
                func_name_called = tool_call.function.name
                args = json.loads(tool_call.function.arguments or "{}")
                if func_name_called == "mark_mapping_valid":
                    return True, None
                if func_name_called == "mark_mapping_ambiguous":
                    return True, args.get("reason", "")
                if func_name_called == "mark_mapping_invalid":
                    return False, args.get("reason", "Invalid mapping")
            
            return False, "No response from 2nd validator"
            
        except OpenAIAPIError:
            raise
        except Exception as e:
            logger.error(f"2nd validation error: {str(e)}")
            return False, f"Validation error: {str(e)}"

    def validate_multi(self, utterance: str, function_calls: List[Dict]) -> Tuple[bool, Optional[str]]:
        """
        2nd stage validation for a full multi-function mapping.
        """
        try:
            normalized_calls = [_normalize_call(call) for call in function_calls]
            calls_str = json.dumps(normalized_calls, ensure_ascii=False)
            prompt = f"""Analyze if this user utterance correctly maps to the full function-call set.

Utterance: "{utterance}"
Function calls: {calls_str}

Questions to check:
1. Does the utterance include every listed function-call intention?
2. Are required arguments correct?
3. Is the mapping acceptable even if optional arguments like target_date are omitted?
4. Ignore function-call order. The same calls in a different order are still valid.

Call exactly one validation decision function:
- mark_mapping_valid if the full mapping is correct and clear
- mark_mapping_ambiguous if the full mapping is acceptable but ambiguous
- mark_mapping_invalid if the full mapping is wrong
"""
            response = call_openai_with_retry(
                client=self.client,
                model=MODEL_NAME,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                tools=self._build_validation_schema(),
                tool_choice="required"
            )

            tool_calls = response.choices[0].message.tool_calls or []
            if tool_calls:
                tool_call = tool_calls[0]
                func_name_called = tool_call.function.name
                args = json.loads(tool_call.function.arguments or "{}")
                if func_name_called == "mark_mapping_valid":
                    return True, None
                if func_name_called == "mark_mapping_ambiguous":
                    return True, args.get("reason", "")
                if func_name_called == "mark_mapping_invalid":
                    return False, args.get("reason", "Invalid mapping")

            return False, "No response from 2nd validator"

        except OpenAIAPIError:
            raise
        except Exception as e:
            logger.error(f"2nd multi validation error: {str(e)}")
            return False, f"Validation error: {str(e)}"


class ValidationPipeline:
    """Main validation pipeline"""
    
    def __init__(self):
        self.validator_1st = OpenAIValidator1st()
        self.validator_2nd = OpenAIValidator2nd()
        self.rule_validator = RuleBasedValidator()
        
        self.validated_data = []
        self.rejected_data = []
    
    def validate_sample(self, row: pd.Series, data_type: str) -> Tuple[bool, Optional[str], Optional[str], Optional[Dict], Optional[str]]:
        """
        Validate a single sample through 2-stage validation
        Returns: (is_valid, reject_stage, ambiguity_note, validator_prediction, api_error_reason)
        """
        utterance = row['user_query']

        if data_type == "multi" and "function_calls" in row and pd.notna(row.get("function_calls")):
            try:
                ground_truth_calls = (
                    json.loads(row["function_calls"])
                    if isinstance(row["function_calls"], str)
                    else row["function_calls"]
                )
            except Exception:
                return False, "parsing_error", None, None, None

            if not isinstance(ground_truth_calls, list) or not ground_truth_calls:
                return False, "parsing_error", None, None, None

            try:
                predicted_calls, is_valid_1st = self.validator_1st.validate_multi(utterance)
            except OpenAIAPIError as e:
                return False, "api_error", None, None, str(e)

            predicted_payload = {"function_calls": predicted_calls}
            if not is_valid_1st or not predicted_calls:
                return False, "1st_validation", None, predicted_payload, None

            if not self.validator_1st.exact_match_many(predicted_calls, ground_truth_calls):
                return False, "1st_validation", None, predicted_payload, None

            try:
                is_valid_2nd, ambiguity_desc = self.validator_2nd.validate_multi(
                    utterance, predicted_calls
                )
            except OpenAIAPIError as e:
                return True, None, f"2nd_validator_api_error: {str(e)}", predicted_payload, None

            if not is_valid_2nd:
                reason = ambiguity_desc or "invalid_or_uncertain"
                return True, None, f"2nd_validator_flagged: {reason}", predicted_payload, None

            for predicted in predicted_calls:
                normalized = _normalize_call(predicted)
                warnings = self.rule_validator.check_warning_rules(
                    utterance, normalized["function_name"], normalized["arguments"]
                )
                for warning in warnings:
                    warning_logger.warning(warning)

            return True, None, ambiguity_desc, predicted_payload, None

        ground_truth_func = row['function_name']
        ground_truth_args = row.get('arguments', {})
        
        # Parse ground truth arguments
        if isinstance(ground_truth_args, str):
            try:
                ground_truth_args = json.loads(ground_truth_args)
            except:
                ground_truth_args = {}
        
        ground_truth = {
            "function_name": ground_truth_func,
            "arguments": ground_truth_args
        }
        
        # 1st stage validation: Function calling
        try:
            predicted, is_valid_1st = self.validator_1st.validate(utterance)
        except OpenAIAPIError as e:
            return False, "api_error", None, None, str(e)
        
        if not is_valid_1st or not predicted:
            return False, "1st_validation", None, predicted, None
        
        # Check exact match
        if not self.validator_1st.exact_match(predicted, ground_truth):
            return False, "1st_validation", None, predicted, None
        
        validation_args = _required_arguments_only(
            predicted["function_name"], predicted.get("arguments", {})
        )
        
        # 2nd stage validation: Semantic check
        try:
            is_valid_2nd, ambiguity_desc = self.validator_2nd.validate(
                utterance, predicted["function_name"], validation_args
            )
        except OpenAIAPIError as e:
            return False, "api_error", None, predicted, str(e)
        
        if not is_valid_2nd:
            return False, "2nd_validation", ambiguity_desc, predicted, None
        
        # Rule-based warnings
        warnings = self.rule_validator.check_warning_rules(
            utterance, predicted["function_name"], validation_args
        )
        for warning in warnings:
            warning_logger.warning(warning)
        
        return True, None, ambiguity_desc, predicted, None
    
    def validate(self, data_type: str = "single") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Main validation pipeline
        Returns: (validated_df, rejected_df)
        """
        # Read generated data
        if data_type == "single":
            input_path = GENERATED_SINGLE_PATH
            validated_path = VALIDATED_SINGLE_PATH
            rejected_path = REJECTED_SINGLE_PATH
            api_error_path = API_ERROR_SINGLE_PATH
        else:
            input_path = GENERATED_MULTI_PATH
            validated_path = VALIDATED_MULTI_PATH
            rejected_path = REJECTED_MULTI_PATH
            api_error_path = API_ERROR_MULTI_PATH
        
        if not Path(input_path).exists():
            logger.error(f"Generated file not found: {input_path}")
            return pd.DataFrame(), pd.DataFrame()
        
        generated_df = pd.read_csv(input_path)
        logger.info(f"Starting validation of {len(generated_df)} samples ({data_type}-turn)")
        
        validated_samples = []
        rejected_samples = []
        api_error_samples = []
        
        for idx, row in generated_df.iterrows():
            is_valid, reject_stage, ambiguity, predicted, api_error_reason = self.validate_sample(row, data_type)
            
            if is_valid:
                if data_type == "multi":
                    validated_row = {
                        "id": row.get('id', idx),
                        "user_query": row['user_query'],
                        "function_calls": row.get('function_calls', '[]'),
                        "ambiguous": ambiguity if ambiguity else ""
                    }
                else:
                    required_args = _required_arguments_only(
                        row['function_name'], predicted.get("arguments", {}) if predicted else {}
                    )
                    validated_row = {
                        "id": row.get('id', idx),
                        "user_query": row['user_query'],
                        "function_name": row['function_name'],
                        "arguments": json.dumps(required_args, ensure_ascii=False),
                        "ambiguous": ambiguity if ambiguity else ""
                    }
                validated_samples.append(validated_row)
                logger.debug(f"✓ [{idx+1}/{len(generated_df)}] VALID")
            elif reject_stage == "api_error":
                if data_type == "multi":
                    api_error_row = {
                        "id": row.get('id', idx),
                        "user_query": row['user_query'],
                        "function_calls": row.get('function_calls', '[]'),
                        "error_reason": api_error_reason if api_error_reason else "unknown api error"
                    }
                else:
                    api_error_row = {
                        "id": row.get('id', idx),
                        "user_query": row['user_query'],
                        "function_name": row.get('function_name', ''),
                        "arguments": row.get('arguments', '{}'),
                        "error_reason": api_error_reason if api_error_reason else "unknown api error"
                    }
                api_error_samples.append(api_error_row)
                logger.debug(f"⚠ [{idx+1}/{len(generated_df)}] API ERROR")
            else:
                # Add to rejected
                if data_type == "multi":
                    rejected_row = {
                        "id": row.get('id', idx),
                        "user_query": row['user_query'],
                        "function_calls": row.get('function_calls', '[]'),
                        "validator_pred": json.dumps(predicted or {}, ensure_ascii=False),
                        "reject_stage": reject_stage if reject_stage else "unknown"
                    }
                else:
                    rejected_row = {
                        "id": row.get('id', idx),
                        "user_query": row['user_query'],
                        "function_name": row.get('function_name', ''),
                        "arguments": row.get('arguments', '{}'),
                        "validator_pred": json.dumps(predicted or {}, ensure_ascii=False),
                        "reject_stage": reject_stage if reject_stage else "unknown"
                    }
                rejected_samples.append(rejected_row)
                logger.debug(f"✗ [{idx+1}/{len(generated_df)}] REJECTED ({reject_stage})")
        
        # Create DataFrames
        if data_type == "multi":
            validated_columns = ["id", "user_query", "function_calls", "ambiguous"]
            rejected_columns = ["id", "user_query", "function_calls", "validator_pred", "reject_stage"]
            api_error_columns = ["id", "user_query", "function_calls", "error_reason"]
        else:
            validated_columns = ["id", "user_query", "function_name", "arguments", "ambiguous"]
            rejected_columns = ["id", "user_query", "function_name", "arguments", "validator_pred", "reject_stage"]
            api_error_columns = ["id", "user_query", "function_name", "arguments", "error_reason"]
        validated_df = pd.DataFrame(validated_samples, columns=validated_columns)
        rejected_df = pd.DataFrame(rejected_samples, columns=rejected_columns)
        api_error_df = pd.DataFrame(api_error_samples, columns=api_error_columns)
        
        # Save to CSV
        if Path(validated_path).exists() and Path(validated_path).stat().st_size > 1:
            existing_df = pd.read_csv(validated_path)
            validated_df = pd.concat([existing_df, validated_df], ignore_index=True)
        validated_df.to_csv(validated_path, index=False)
        
        if Path(rejected_path).exists() and Path(rejected_path).stat().st_size > 1:
            existing_df = pd.read_csv(rejected_path)
            rejected_df = pd.concat([existing_df, rejected_df], ignore_index=True)
        rejected_df.to_csv(rejected_path, index=False)

        if Path(api_error_path).exists() and Path(api_error_path).stat().st_size > 1:
            existing_df = pd.read_csv(api_error_path)
            api_error_df = pd.concat([existing_df, api_error_df], ignore_index=True)
        api_error_df.to_csv(api_error_path, index=False)
        
        # Summary
        logger.info("=" * 80)
        logger.info(f"VALIDATION RESULT ({data_type}-turn)")
        logger.info("=" * 80)
        logger.info(f"Total: {len(generated_df)}")
        if len(generated_df) > 0:
            logger.info(f"✓ Valid: {len(validated_df)} ({100*len(validated_df)/len(generated_df):.1f}%)")
            logger.info(f"✗ Rejected: {len(rejected_df)} ({100*len(rejected_df)/len(generated_df):.1f}%)")
            logger.info(f"⚠ API Error: {len(api_error_df)} ({100*len(api_error_df)/len(generated_df):.1f}%)")
        else:
            logger.info("✓ Valid: 0 (0.0%)")
            logger.info("✗ Rejected: 0 (0.0%)")
            logger.info("⚠ API Error: 0 (0.0%)")
        
        if len(rejected_df) > 0:
            reject_stage_counts = rejected_df['reject_stage'].value_counts()
            for stage, count in reject_stage_counts.items():
                logger.info(f"  - {stage}: {count}")
        
        logger.info("=" * 80)
        
        return validated_df, rejected_df


def main():
    """Main validation pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate generated mission data")
    parser.add_argument("--mode", choices=["single", "multi", "all"], default="all",
                        help="Data type to validate")
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting Validation Pipeline")
    
    pipeline = ValidationPipeline()
    
    if args.mode in ["single", "all"]:
        logger.info("\n" + "="*80)
        logger.info("VALIDATING SINGLE-TURN DATA")
        logger.info("="*80)
        pipeline.validate(data_type="single")
    
    if args.mode in ["multi", "all"]:
        logger.info("\n" + "="*80)
        logger.info("VALIDATING MULTI-TURN DATA")
        logger.info("="*80)
        pipeline.validate(data_type="multi")
    
    logger.info("\n✨ Validation pipeline completed!")


if __name__ == "__main__":
    main()
