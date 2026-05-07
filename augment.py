"""
Data augmentation module: Generate new utterances using OpenAI
with Few-shot, Contrastive Few-shot, and style diversity
"""

import os
import json
import logging
import time
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from openai import OpenAI

from config import (
    MODEL_NAME, OPENAI_API_KEY, TOOLS, TARGETS_SINGLE, TARGETS_MULTI, EXTRA_HINTS,
    SEED_SINGLE_PATH, SEED_MULTI_PATH,
    GENERATED_SINGLE_PATH, GENERATED_MULTI_PATH,
    DATA_DIR, GENERATION_MULTIPLIER, MAX_GENERATION_ROUNDS, BATCH_SIZE_PER_ROUND
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{DATA_DIR}/augmentation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


AUGMENT_SYSTEM = (
    "너는 어린이 건강습관 코치 앱의 function calling 학습 데이터셋 생성 전문가야. "
    "초등학생(8~13세) 스타일의 자연스러운 한국어 사용자 발화를 생성하는 역할이야. "
    "함수 호출 JSON이나 설명은 만들지 말고, 주어진 함수/인자 조합에 맞는 발화만 생성해.\n\n"

    "## 사용자 특성 (반드시 반영)\n"
    "- 초등학생(8~13세), 반말/존댓말/혼용 중 하나를 랜덤으로 사용\n"
    "- 줄임말 사용 (넘, 걍, 좀, 완전, 진짜, 쫌 등)\n"
    "- 감정 표현 포함 (ㅠ, ㅠㅠ, ㅋㅋ, ㅎㅎ, !! 등)\n\n"

    "## 문장 길이 다양성 (반드시 섞기)\n"
    "- 짧은 문장: '못 함', '이거 뭐임', '취소요', '다 했어'\n"
    "- 중간 문장: '오늘 좀 힘들어서 못 했어요ㅠ', '미션 난이도 낮춰줘'\n"
    "- 긴 문장: '오늘 학원 갔다 오니까 너무 늦어서 미션 못 할 것 같아요ㅠ'\n\n"

    "## 현실성 규칙\n"
    "- 애매하게 말하는 표현은 전체의 20~30% 이하로만 포함\n"
    "  (예: '거의 했는데 내도 됨?', '좀 했는데 다는 못 했어요')\n"
    "- 같은 의도도 다양하게 표현할 것\n\n"

    "## 경계 케이스 주의 (혼동 방지)\n"
    "- submit[success]: 완전히 다 했다는 표현만. '반만 했어'는 fail임\n"
    "- submit[fail]: 못 했거나 일부만 한 표현. '다 했어'는 success임\n"
    "- easier: 쉽게 해달라는 것. '다른 걸로 바꿔줘'는 change임\n"
    "- harder: 어렵게 해달라는 것. '다른 걸로 바꿔줘'는 change임\n"
    "- behavior: 행동 대체 (자전거 대신 줄넘기). 장소 바꾸는 건 place임\n"
    "- place: 장소 대체 (집에서, 실내에서). 행동 바꾸는 건 behavior임\n\n"

    "## 스타일 예시\n"
    "- 미션 완료: '다 함ㅋ', '오늘 미션 완료!', '겨우 다 했어요ㅎㅎ'\n"
    "- 미션 실패: '오늘 그냥 패스', '아파서 못 함', '반만 했어 걍 내줘'\n"
    "- 난이도 조정: '쌤 이거 넘 어려워요ㅠ', '너무 쉬움 더 어렵게 해줘'\n"
    "- 취소: '아 잠깐 취소요!', '방금 제출한 거 취소해줘요'\n"
    "- 인정 여부: '줄넘기 대신 자전거 타면 돼요?', '집에서 해도 인정돼요?'\n"
    "- 기록 확인: '나 이번 주 몇 번 했어요?', '이번 달 기록 보여줘요'\n\n"

    "## 금지\n"
    "- 비속어 금지\n"
    "- 영어/일본어 등 외국어 금지\n"
    "- 문장 끝 마침표 격식체 금지 ('미션을 변경해 주세요.' 같은 형태)\n"
    "- 기존 예시와 동일한 문장 금지\n\n"

    "## 출력 형식\n"
    "반드시 JSON 배열만 출력. 설명이나 다른 텍스트 없이:\n"
    "[\"발화1\", \"발화2\", ...]\n"
)


def _tool_properties(func_name: str) -> Dict:
    return TOOLS[func_name].get("parameters", {}).get("properties", {})


def _tool_required(func_name: str) -> List[str]:
    return TOOLS[func_name].get("parameters", {}).get("required", [])


def _target_arg_name(func_name: str) -> str | None:
    required = _tool_required(func_name)
    return required[0] if required else None


def _call_arg_value(func_name: str, arguments: Dict) -> str | None:
    arg_name = _target_arg_name(func_name)
    if not arg_name:
        return None
    return (arguments or {}).get(arg_name)


def _build_function_call(func_name: str, arg_value: str | None) -> Dict:
    arg_name = _target_arg_name(func_name)
    arguments = {} if not arg_name or arg_value is None else {arg_name: arg_value}
    return {"name": func_name, "arguments": arguments}


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
    raise Exception(f"최대 재시도 횟수 초과: {last_error}")


class GapAnalyzer:
    """Analyze data gap for each function-argument combination"""
    
    def __init__(self, seed_single_path: str, seed_multi_path: str):
        self.seed_single_df = pd.read_csv(seed_single_path)
        self.seed_multi_df = pd.read_csv(seed_multi_path)
        self.gap_analysis = {}
    
    def analyze(self) -> Dict[str, int]:
        """
        Analyze current data distribution and calculate gap for each function-arg combination
        Returns: {(function_name, arg_value): need_count, ...}
        """
        # Parse single-turn data
        single_turn_counts = {}
        for _, row in self.seed_single_df.iterrows():
            func_name = row['function_name']
            args = json.loads(row['arguments']) if isinstance(row['arguments'], str) else row['arguments']
            
            # For functions with arguments, track each argument value
            if args:
                for key, value in args.items():
                    key_tuple = (func_name, key, value)
                    single_turn_counts[key_tuple] = single_turn_counts.get(key_tuple, 0) + 1
            else:
                # For functions without arguments
                key_tuple = (func_name, None, None)
                single_turn_counts[key_tuple] = single_turn_counts.get(key_tuple, 0) + 1
        
        # Parse multi-turn data
        multi_turn_counts = {}
        for _, row in self.seed_multi_df.iterrows():
            function_calls = json.loads(row['function_calls']) if isinstance(row['function_calls'], str) else row['function_calls']
            
            for call in function_calls:
                func_name = call['name']
                args = call.get('arguments', {})
                
                if args:
                    for key, value in args.items():
                        key_tuple = (func_name, key, value)
                        multi_turn_counts[key_tuple] = multi_turn_counts.get(key_tuple, 0) + 1
                else:
                    key_tuple = (func_name, None, None)
                    multi_turn_counts[key_tuple] = multi_turn_counts.get(key_tuple, 0) + 1
        
        # Calculate gap
        gap_dict = {}
        
        for func_name, arg_options in TARGETS_SINGLE.items():
            for arg_value, target_count in arg_options.items():
                # Handle cancel_mission_action (no arguments)
                if arg_value == "no_args":
                    current_single = single_turn_counts.get((func_name, None, None), 0)
                    current_multi = multi_turn_counts.get((func_name, None, None), 0)
                    current_total = current_single + current_multi
                else:
                    # Extract argument name (first key in the argument dict)
                    arg_name = _target_arg_name(func_name)
                    
                    current_single = single_turn_counts.get((func_name, arg_name, arg_value), 0)
                    current_multi = multi_turn_counts.get((func_name, arg_name, arg_value), 0)
                    current_total = current_single + current_multi
                
                need = max(0, target_count - current_total)
                
                key = f"{func_name}:{arg_value}"
                gap_dict[key] = {
                    "current": current_total,
                    "target": target_count,
                    "need": need,
                    "single": current_single,
                    "multi": current_multi
                }
        
        logger.info("=" * 80)
        logger.info("GAP ANALYSIS RESULT")
        logger.info("=" * 80)
        for key, stats in gap_dict.items():
            if stats["need"] > 0:
                logger.info(
                    f"{key:50s} | Current: {stats['current']:3d} | "
                    f"Target: {stats['target']:3d} | Need: {stats['need']:3d}"
                )
        logger.info("=" * 80)
        
        self.gap_analysis = gap_dict
        return gap_dict

    def analyze_multi(self) -> Dict[Tuple[Tuple[str, str | None], ...], Dict[str, int]]:
        """
        Analyze current multi-function pair distribution.
        Returns: {target_pair: {"current": int, "target": int, "need": int}, ...}
        """
        current_counts = {}

        for _, row in self.seed_multi_df.iterrows():
            try:
                function_calls = (
                    json.loads(row["function_calls"])
                    if isinstance(row["function_calls"], str)
                    else row["function_calls"]
                )
            except Exception:
                continue

            if not isinstance(function_calls, list) or len(function_calls) < 2:
                continue

            pair = []
            for call in function_calls[:2]:
                func_name = call.get("name")
                if not func_name:
                    continue
                arg_value = _call_arg_value(func_name, call.get("arguments", {}))
                pair.append((func_name, arg_value))

            if len(pair) == 2:
                key = tuple(pair)
                current_counts[key] = current_counts.get(key, 0) + 1

        gap_dict = {}
        for target_pair, target_count in TARGETS_MULTI.items():
            current = current_counts.get(target_pair, 0)
            gap_dict[target_pair] = {
                "current": current,
                "target": target_count,
                "need": max(0, target_count - current),
            }

        logger.info("=" * 80)
        logger.info("MULTI GAP ANALYSIS RESULT")
        logger.info("=" * 80)
        for key, stats in gap_dict.items():
            if stats["need"] > 0:
                label = " + ".join(f"{func}:{arg}" for func, arg in key)
                logger.info(
                    f"{label:80s} | Current: {stats['current']:3d} | "
                    f"Target: {stats['target']:3d} | Need: {stats['need']:3d}"
                )
        logger.info("=" * 80)

        self.gap_analysis = gap_dict
        return gap_dict


class AugmentationEngine:
    """Generate utterances using OpenAI with Few-shot and Contrastive learning"""
    
    def __init__(self, gap_analysis: Dict, seed_single_df: pd.DataFrame, seed_multi_df: pd.DataFrame):
        self.gap_analysis = gap_analysis
        self.seed_single_df = seed_single_df
        self.seed_multi_df = seed_multi_df
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.generated_data = {"single": [], "multi": []}
    
    def _get_few_shot_examples(self, func_name: str, arg_name: str, arg_value: str, 
                               seed_df: pd.DataFrame, previous_generated: List[Dict] = None) -> List[str]:
        """
        Build few-shot examples including contrastive pairs
        """
        examples = []
        
        # Get seed examples for this function-argument combination.
        # Single-turn seeds have function_name/arguments; multi-turn seeds have function_calls.
        for _, row in seed_df.iterrows():
            if self._row_matches_target(row, func_name, arg_value):
                examples.append(row['user_query'])
            if len(examples) >= 2:
                break
        seed_examples = examples
        
        few_shot_text = "## Few-shot Examples\n"
        
        # Add seed examples
        for i, ex in enumerate(seed_examples, 1):
            few_shot_text += f"{i}. {ex}\n"
        
        # Add contrastive examples for every confusion set that contains this value.
        contrastive_examples = self._get_contrastive_examples(arg_value)
        if contrastive_examples:
            few_shot_text += "\n## Contrastive Examples (What NOT to do)\n"
            for i, item in enumerate(contrastive_examples, 1):
                few_shot_text += f"❌ {i}. [{item['label']}] {item['example']}\n"
        
        # Add previously generated examples to maintain diversity
        if previous_generated:
            few_shot_text += "\n## Previous Generated (for diversity)\n"
            for i, item in enumerate(previous_generated[-3:], 1):  # Last 3 generated
                few_shot_text += f"{i}. {item.get('user_query', '')}\n"
        
        return few_shot_text

    def _get_contrastive_examples(self, arg_value: str) -> List[Dict[str, str]]:
        """Collect opposite-class examples from every configured contrastive hint set."""
        contrastive_examples = []
        
        for hint_name, hint in EXTRA_HINTS.items():
            good_examples = hint.get("good_examples", {})
            if arg_value not in good_examples:
                continue
            
            for label, examples in good_examples.items():
                if label == arg_value:
                    continue
                for example in examples[:2]:
                    contrastive_examples.append({
                        "hint": hint_name,
                        "label": label,
                        "example": example
                    })
        
        return contrastive_examples
    
    def _row_matches_target(self, row: pd.Series, func_name: str, arg_value: str) -> bool:
        """Return True if a seed row contains the target function/argument value."""
        if 'function_calls' in row and pd.notna(row.get('function_calls')):
            try:
                function_calls = json.loads(row['function_calls']) if isinstance(row['function_calls'], str) else row['function_calls']
            except Exception:
                function_calls = []
            for call in function_calls:
                if call.get('name') != func_name:
                    continue
                args = call.get('arguments', {})
                if arg_value == "no_args":
                    return not args
                if arg_value in args.values():
                    return True
            return False

        if row.get('function_name') != func_name:
            return False

        try:
            args = json.loads(row['arguments']) if isinstance(row.get('arguments'), str) else row.get('arguments', {})
        except Exception:
            args = {}

        if arg_value == "no_args":
            return not args
        return arg_value in args.values()
    
    def generate_batch(self, func_name: str, arg_name: str, arg_value: str, 
                       need_count: int, batch_index: int, 
                       seed_df: pd.DataFrame, previous_generated: List[Dict] = None) -> List[Dict]:
        """
        Generate a batch of utterances for specific function-argument combination
        """
        generation_count = int(need_count * GENERATION_MULTIPLIER)
        batch_size = BATCH_SIZE_PER_ROUND
        samples_per_batch = generation_count // batch_size
        remainder = generation_count % batch_size
        
        logger.info(f"\n🔄 Generating for {func_name}:{arg_value}")
        logger.info(f"   Need: {need_count} | Generate: {generation_count} | Batches: {samples_per_batch + (1 if remainder else 0)}")
        
        generated_samples = []
        previous_generated = previous_generated or []
        
        # Generate in multiple rounds
        for round_idx in range(samples_per_batch + (1 if remainder else 0)):
            current_batch_size = BATCH_SIZE_PER_ROUND if round_idx < samples_per_batch else remainder
            
            # Get few-shot examples
            few_shot_examples = self._get_few_shot_examples(
                func_name, arg_name, arg_value, seed_df, previous_generated + generated_samples
            )
            
            # Build prompt
            prompt = self._build_prompt(
                func_name, arg_name, arg_value, current_batch_size,
                few_shot_examples, batch_index, round_idx
            )
            
            try:
                # Generate utterance text only. Function classification is handled in validate.py.
                response = call_openai_with_retry(
                    client=self.client,
                    model=MODEL_NAME,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.9
                )
                
                # Parse response
                samples = self._parse_response(response, func_name, arg_name, arg_value, round_idx)
                generated_samples.extend(samples)
                
                logger.info(f"   ✓ Round {round_idx + 1}: Generated {len(samples)} samples")
                
            except Exception as e:
                logger.error(f"   ✗ Error generating batch: {str(e)}")
                continue
        
        return generated_samples
    
    def _build_prompt(self, func_name: str, arg_name: str, arg_value: str,
                      batch_size: int, few_shot_examples: str, 
                      batch_index: int, round_idx: int) -> str:
        """Build the augmentation prompt"""
        
        if arg_value == "no_args":
            prompt = f"""{AUGMENT_SYSTEM}

## 생성 대상

사용자의 자연스러운 발화를 생성해야 합니다.
- 함수: {func_name}
- 이 함수는 특별한 인자가 필요 없습니다.

{few_shot_examples}

## Task
위의 예시와 유사한 스타일로, **정확히 {batch_size}개**의 자연스러운 한국어 발화를 생성하세요.
각 발화는 사용자가 {func_name}를 호출하고자 하는 의도를 명확하게 나타내야 합니다.

다양한 표현과 뉘앙스를 사용하되, 모든 발화는 함수 호출로 {func_name}를 예측하도록 만들어야 합니다.

응답은 반드시 JSON 배열만 출력하세요.
형식: ["발화1", "발화2", ...]
"""
        else:
            # Get argument enum values
            arg_enum = _tool_properties(func_name).get(arg_name, {}).get("enum", [])
            others = [v for v in arg_enum if v != arg_value]
            
            prompt = f"""{AUGMENT_SYSTEM}

## 생성 대상

사용자의 자연스러운 발화를 생성해야 합니다.
- 함수: {func_name}
- 인자: {arg_name} = {arg_value}
- 주의: 절대로 {arg_name}을(를) {', '.join(others)}로 설정하면 안 됩니다.

{few_shot_examples}

## Task
위의 예시와 유사한 스타일로, **정확히 {batch_size}개**의 자연스러운 한국어 발화를 생성하세요.
각 발화는 사용자가 {func_name}(파라미터: {arg_value})를 호출하고자 하는 의도를 명확하게 나타내야 합니다.

다양한 표현과 뉘앙스를 사용하되, 모든 발화는 함수 호출로 {func_name}({arg_name}={arg_value})를 예측하도록 만들어야 합니다.

응답은 반드시 JSON 배열만 출력하세요.
형식: ["발화1", "발화2", ...]
"""
        
        return prompt
    
    def _parse_response(self, response, func_name: str, arg_name: str, arg_value: str, round_idx: int) -> List[Dict]:
        """Parse OpenAI response and extract generated samples"""
        samples = []
        
        try:
            result_text = response.choices[0].message.content.strip()
            utterances = json.loads(result_text)
            if isinstance(utterances, dict):
                utterances = utterances.get("utterances", [])
            if not isinstance(utterances, list):
                logger.error("OpenAI response was not a JSON array")
                return samples

            args_dict = {} if arg_value == "no_args" else {arg_name: arg_value}
            for utterance in utterances:
                if not isinstance(utterance, str) or not utterance.strip():
                    continue
                samples.append({
                    "user_query": utterance.strip(),
                    "function_name": func_name,
                    "arguments": json.dumps(args_dict, ensure_ascii=False),
                    "generation_round": round_idx
                })
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
        
        return samples

    def _row_matches_pair(self, row: pd.Series, target_pair: Tuple[Tuple[str, str | None], ...]) -> bool:
        """Return True if a seed multi row exactly matches the target ordered pair."""
        if "function_calls" not in row or pd.isna(row.get("function_calls")):
            return False
        try:
            function_calls = (
                json.loads(row["function_calls"])
                if isinstance(row["function_calls"], str)
                else row["function_calls"]
            )
        except Exception:
            return False
        if not isinstance(function_calls, list) or len(function_calls) < len(target_pair):
            return False

        normalized = []
        for call in function_calls[:len(target_pair)]:
            func_name = call.get("name")
            normalized.append((func_name, _call_arg_value(func_name, call.get("arguments", {}))))
        return tuple(normalized) == target_pair

    def _get_multi_few_shot_examples(
        self,
        target_pair: Tuple[Tuple[str, str | None], ...],
        previous_generated: List[Dict] = None,
    ) -> str:
        """Build few-shot examples for ordered multi-function pairs."""
        examples = []
        for _, row in self.seed_multi_df.iterrows():
            if self._row_matches_pair(row, target_pair):
                examples.append(row["user_query"])
            if len(examples) >= 2:
                break

        few_shot_text = "## Few-shot Examples\n"
        for i, ex in enumerate(examples, 1):
            few_shot_text += f"{i}. {ex}\n"

        if previous_generated:
            few_shot_text += "\n## Previous Generated (for diversity)\n"
            for i, item in enumerate(previous_generated[-3:], 1):
                few_shot_text += f"{i}. {item.get('user_query', '')}\n"

        return few_shot_text

    def _build_multi_prompt(
        self,
        target_pair: Tuple[Tuple[str, str | None], ...],
        batch_size: int,
        few_shot_examples: str,
    ) -> str:
        call_lines = []
        for idx, (func_name, arg_value) in enumerate(target_pair, 1):
            arg_name = _target_arg_name(func_name)
            if arg_name and arg_value is not None:
                call_lines.append(f"{idx}. {func_name}({arg_name}={arg_value})")
            else:
                call_lines.append(f"{idx}. {func_name}()")

        return f"""{AUGMENT_SYSTEM}

## 생성 대상

사용자 한 문장 안에 아래 두 의도가 모두 자연스럽게 들어간 발화를 생성해야 합니다.
함수 호출 JSON은 출력하지 말고, 발화 문자열만 생성하세요.

{chr(10).join(call_lines)}

{few_shot_examples}

## Multi-function 규칙
- 한 발화 안에 위 1번과 2번 의도가 모두 포함되어야 합니다.
- 두 의도 중 하나만 말하는 발화는 만들지 마세요.
- 너무 기계적으로 "그리고"만 반복하지 말고 자연스러운 초등학생 말투로 섞으세요.
- "오늘 미션 뭐야/알려줘/보여줘"는 get_mission_info(query_type=today) 의도입니다. 기록 조회가 아닙니다.
- "이번 주/이번 달/오늘/어제 기록, 몇 번 했는지, 성공/실패 기록"은 get_user_history 의도입니다. 미션 정보 조회가 아닙니다.
- "미션 쉽게/어렵게 해줘"와 "오늘 미션 알려줘"가 같이 있으면 request_mission_adjustment + get_mission_info 조합으로 분명히 표현하세요.
- "집에서/실내에서 해도 돼?"는 place, "자전거/줄넘기/춤으로 해도 돼?"는 behavior로 분명히 표현하세요.

## Task
위 조건에 맞는 자연스러운 한국어 발화를 **정확히 {batch_size}개** 생성하세요.
응답은 반드시 JSON 배열만 출력하세요.
형식: ["발화1", "발화2", ...]
"""

    def _parse_multi_response(
        self,
        response,
        target_pair: Tuple[Tuple[str, str | None], ...],
        round_idx: int,
    ) -> List[Dict]:
        samples = []
        try:
            result_text = response.choices[0].message.content.strip()
            utterances = json.loads(result_text)
            if isinstance(utterances, dict):
                utterances = utterances.get("utterances", [])
            if not isinstance(utterances, list):
                logger.error("OpenAI response was not a JSON array")
                return samples

            function_calls = [
                _build_function_call(func_name, arg_value)
                for func_name, arg_value in target_pair
            ]
            function_calls_json = json.dumps(function_calls, ensure_ascii=False)

            for utterance in utterances:
                if not isinstance(utterance, str) or not utterance.strip():
                    continue
                samples.append({
                    "user_query": utterance.strip(),
                    "function_calls": function_calls_json
                })
        except Exception as e:
            logger.error(f"Error parsing multi response: {str(e)}")

        return samples

    def generate_multi_batch(
        self,
        target_pair: Tuple[Tuple[str, str | None], ...],
        need_count: int,
        previous_generated: List[Dict] = None,
    ) -> List[Dict]:
        """Generate utterances for one ordered multi-function target pair."""
        generation_count = int(need_count * GENERATION_MULTIPLIER)
        batch_size = BATCH_SIZE_PER_ROUND
        samples_per_batch = generation_count // batch_size
        remainder = generation_count % batch_size
        label = " + ".join(f"{func}:{arg}" for func, arg in target_pair)

        logger.info(f"\n🔄 Generating for {label}")
        logger.info(f"   Need: {need_count} | Generate: {generation_count} | Batches: {samples_per_batch + (1 if remainder else 0)}")

        generated_samples = []
        previous_generated = previous_generated or []

        for round_idx in range(samples_per_batch + (1 if remainder else 0)):
            current_batch_size = BATCH_SIZE_PER_ROUND if round_idx < samples_per_batch else remainder
            few_shot_examples = self._get_multi_few_shot_examples(
                target_pair, previous_generated + generated_samples
            )
            prompt = self._build_multi_prompt(target_pair, current_batch_size, few_shot_examples)

            try:
                response = call_openai_with_retry(
                    client=self.client,
                    model=MODEL_NAME,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.9
                )
                samples = self._parse_multi_response(response, target_pair, round_idx)
                generated_samples.extend(samples)
                logger.info(f"   ✓ Round {round_idx + 1}: Generated {len(samples)} samples")
            except Exception as e:
                logger.error(f"   ✗ Error generating multi batch: {str(e)}")
                continue

        return generated_samples

    def _augment_multi(self) -> pd.DataFrame:
        all_generated = []

        for target_pair, stats in self.gap_analysis.items():
            if stats["need"] <= 0:
                label = " + ".join(f"{func}:{arg}" for func, arg in target_pair)
                logger.info(f"⊘ Skipping {label} (no need)")
                continue

            batch = self.generate_multi_batch(target_pair, stats["need"])
            all_generated.extend(batch)

        generated_df = pd.DataFrame(all_generated, columns=[
            "user_query", "function_calls"
        ])

        existing_count = len(self.seed_multi_df)
        generated_df.insert(0, "id", range(existing_count + 1, existing_count + len(generated_df) + 1))

        output_path = GENERATED_MULTI_PATH
        if Path(output_path).exists() and Path(output_path).stat().st_size > 3:
            existing_df = pd.read_csv(output_path)
            final_df = pd.concat([existing_df, generated_df], ignore_index=True)
        else:
            final_df = generated_df

        final_df.to_csv(output_path, index=False)
        logger.info(f"\n✅ Saved {len(generated_df)} new multi samples to {output_path}")
        logger.info(f"   Total samples: {len(final_df)}")

        return final_df
    
    def augment(self, data_type: str = "single") -> pd.DataFrame:
        """
        Main augmentation pipeline
        data_type: "single" or "multi"
        """
        if data_type == "multi":
            return self._augment_multi()

        if data_type == "single":
            seed_df = self.seed_single_df
            output_path = GENERATED_SINGLE_PATH
        else:
            seed_df = self.seed_multi_df
            output_path = GENERATED_MULTI_PATH
        
        all_generated = []
        
        # Skip cancel_mission_action for multi-turn (usually single-turn action)
        skip_list = [] if data_type == "single" else ["cancel_mission_action"]
        
        for key_str, stats in self.gap_analysis.items():
            if stats["need"] <= 0:
                logger.info(f"⊘ Skipping {key_str} (no need)")
                continue
            
            func_name, arg_value = key_str.split(":")
            
            if func_name in skip_list:
                logger.info(f"⊘ Skipping {key_str} (not applicable for {data_type}-turn)")
                continue
            
            # Get argument name
            if arg_value != "no_args":
                arg_name = _target_arg_name(func_name)
            else:
                arg_name = None
            
            # Generate batch
            batch = self.generate_batch(
                func_name, arg_name, arg_value,
                stats["need"], 0, seed_df
            )
            all_generated.extend(batch)
        
        # Create DataFrame
        generated_df = pd.DataFrame(all_generated)
        
        # Add ID column
        existing_count = len(seed_df)
        generated_df.insert(0, 'id', range(existing_count + 1, existing_count + len(generated_df) + 1))
        
        # Save to CSV
        if Path(output_path).exists() and Path(output_path).stat().st_size > 3:
            existing_df = pd.read_csv(output_path)
            final_df = pd.concat([existing_df, generated_df], ignore_index=True)
        else:
            final_df = generated_df
        
        final_df.to_csv(output_path, index=False)
        logger.info(f"\n✅ Saved {len(generated_df)} new samples to {output_path}")
        logger.info(f"   Total samples: {len(final_df)}")
        
        return final_df


def main():
    """Main augmentation pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Augment mission data using OpenAI")
    parser.add_argument("--mode", choices=["single", "multi", "all"], default="all",
                        help="Data type to augment")
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting Data Augmentation Pipeline")
    
    analyzer = GapAnalyzer(SEED_SINGLE_PATH, SEED_MULTI_PATH)
    
    if args.mode in ["single", "all"]:
        logger.info("\n" + "="*80)
        logger.info("AUGMENTING SINGLE-TURN DATA")
        logger.info("="*80)
        gap_analysis = analyzer.analyze()
        engine = AugmentationEngine(gap_analysis, analyzer.seed_single_df, analyzer.seed_multi_df)
        engine.augment(data_type="single")
    
    if args.mode in ["multi", "all"]:
        logger.info("\n" + "="*80)
        logger.info("AUGMENTING MULTI-TURN DATA")
        logger.info("="*80)
        gap_analysis = analyzer.analyze_multi()
        engine = AugmentationEngine(gap_analysis, analyzer.seed_single_df, analyzer.seed_multi_df)
        engine.augment(data_type="multi")
    
    logger.info("\n✨ Augmentation pipeline completed!")


if __name__ == "__main__":
    main()
