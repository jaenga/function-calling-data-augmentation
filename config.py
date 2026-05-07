"""
Configuration file for mission data augmentation & validation pipeline
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# OPENAI API CONFIGURATION
# ============================================================================
MODEL_NAME = "gpt-4o-mini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_KEY = OPENAI_API_KEY  # Backward-compatible alias

# ============================================================================
# TOOLS DEFINITION
# ============================================================================
# 6 mission-related functions with production parameter schemas
TOOLS = {
    "submit_mission_result": {
        "description": (
            "미션 수행 결과를 확정해서 보고할 때 호출합니다.\n"
            "- success: 완료 (다 했어요)\n"
            "- fail: 수행 못함 (못 했어요, 조금 했어요)\n"
            "- 사용하지 않는 경우: 규칙 질문, 미션 변경 요청, 인정 여부 질문"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "result_type": {
                    "type": "string",
                    "enum": ["success", "fail"],
                    "description": "미션 달성 정도 (값: success[완료], fail[수행 실패])",
                },
            },
            "required": ["result_type"],
        },
    },
    "get_mission_info": {
        "description": (
            "미션 관련 정보(오늘 미션, 마감 시간, 규칙)를 물어볼 때 호출합니다.\n"
            "- today: 오늘/어제/그저께/특정 날짜에 배정된 미션의 내용, 수행 방법, 주의사항, 수행 조건을 물을 때\n"
            "- \"오늘 미션 뭐야?\" → query_type='today', target_date='today'\n"
            "- \"어제 미션 뭐였어?\", \"어제 나 뭐 해야 했어?\" → query_type='today', target_date='yesterday'\n"
            "- \"그저께 미션 뭐였어?\" → query_type='today', target_date='day_before_yesterday'\n"
            "- deadline: 제출 마감, 기한을 물을 때\n"
            "- general_rule: 앱/시스템 차원의 제출·인증·판정·운영 규칙을 물을 때\n"
            "- 사용하지 않는 경우: 결과 보고, 미션 변경 요청, 대체 수행 가능 여부 질문"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query_type": {
                    "type": "string",
                    "enum": ["today", "deadline", "general_rule"],
                    "description": "조회할 정보의 종류",
                },
                "target_date": {
                    "type": "string",
                    "description": "조회할 미션 날짜. today, yesterday, day_before_yesterday 또는 YYYY-MM-DD",
                },
            },
            "required": ["query_type"],
        },
    },
    "request_mission_adjustment": {
        "description": (
            "미션을 바꾸거나 난이도를 조정해 달라고 요청할 때 호출합니다.\n"
            "- change: 다른 미션 요청\n"
            "- easier: 더 쉬운 미션 요청\n"
            "- harder: 더 어려운 미션 요청"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "adjustment_type": {
                    "type": "string",
                    "enum": ["change", "easier", "harder"],
                    "description": "미션 내용 조정 종류",
                },
            },
            "required": ["adjustment_type"],
        },
    },
    "check_mission_equivalency": {
        "description": (
            "다른 행동/장소/시간으로 수행해도 인정되는지 물어볼 때 호출합니다.\n"
            "- behavior: 행동 변경 (자전거로 해도 돼요?)\n"
            "- place: 장소 변경 (집에서 해도 돼요?)\n"
            "- time: 시간 변경 (저녁에 하면 되나요?)"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "equivalency_type": {
                    "type": "string",
                    "enum": ["behavior", "place", "time"],
                    "description": "학생이 제안한 대체 행동 내용",
                },
            },
            "required": ["equivalency_type"],
        },
    },
    "get_user_history": {
        "description": (
            "미션 수행 기록을 조회할 때 호출합니다.\n"
            "- daily_summary: 하루 기록 조회\n"
            "- weekly_summary: 주간 기록 조회\n"
            "- monthly_summary: 월간 기록 조회\n"
            "예시:\n"
            "- 이번 주 기록 → query_type='weekly_summary', target_period='this_week'\n"
            "- 지난주 기록 → query_type='weekly_summary', target_period='last_week'\n"
            "- 이번 달 기록 → query_type='monthly_summary', target_period='this_month'\n"
            "- 지난달 기록 → query_type='monthly_summary', target_period='last_month'\n"
            "- 특정 월 기록을 물으면 target_month에 해당 월을 2자리 숫자로 넣습니다.\n"
            "  예: 4월 기록 → query_type='monthly_summary', target_month='04', "
            "5월 기록 → query_type='monthly_summary', target_month='05'\n"
            "- 오늘 기록 → query_type='daily_summary', target_date='today'\n"
            "- 어제 기록 → query_type='daily_summary', target_date='yesterday'\n"
            "- 그저께 기록 → query_type='daily_summary', target_date='day_before_yesterday'\n"
            "- 4월 12일 기록 → query_type='daily_summary', target_date='YYYY-MM-DD'"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query_type": {
                    "type": "string",
                    "enum": ["daily_summary", "weekly_summary", "monthly_summary"],
                    "description": "조회할 기록의 범위",
                },
                "target_period": {
                    "type": "string",
                    "enum": ["this_week", "last_week", "this_month", "last_month"],
                    "description": "주간/월간 조회 대상 기간. 명확할 때만 사용",
                },
                "target_date": {
                    "type": "string",
                    "description": "일간 조회 날짜. today, yesterday, day_before_yesterday 또는 YYYY-MM-DD",
                },
                "target_month": {
                    "type": "string",
                    "description": "월간 조회 월. MM 또는 YYYY-MM",
                },
            },
            "required": ["query_type"],
        },
    },
    "cancel_mission_action": {
        "description": (
            "미션 제출이나 미션 변경을 취소할 때 호출합니다.\n"
            "- cancel_type='submit': 성공/실패/제출/기록 취소\n"
            "- cancel_type='adjustment': 미션 변경 취소\n"
            "- cancel_type='latest': 최근 작업 취소, 되돌려줘처럼 대상이 모호한 경우\n"
            "예: 성공 취소 / 방금 제출 취소 / 바꾼 거 없던 걸로 / 최근 작업 되돌려줘"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "cancel_type": {
                    "type": "string",
                    "enum": ["submit", "adjustment", "latest"],
                    "description": "취소 대상",
                },
            },
        },
    },
}

# ============================================================================
# GENERATION TARGETS
# ============================================================================
# Target count for each single function-argument combination
# Format: {function_name: {argument_value: target_count, ...}, ...}
TARGETS_SINGLE = {
    "submit_mission_result": {
        "success": 30,
        "fail": 30
    },
    "get_mission_info": {
        "today": 24,
        "deadline": 24,
        "general_rule": 24
    },
    "request_mission_adjustment": {
        "change": 24,
        "easier": 24,
        "harder": 24
    },
    "check_mission_equivalency": {
        "behavior": 24,
        "place": 24,
        "time": 24
    },
    "get_user_history": {
        "weekly_summary": 24,
        "monthly_summary": 24,
        "daily_summary": 24
    },
    "cancel_mission_action": {
        "no_args": 18
    }
}

# Target count for each multi-function ordered pair.
# Format: {((function_name, required_arg_value_or_None), ...): target_count}
TARGETS_MULTI = {
    (("get_mission_info", "today"), ("get_user_history", "weekly_summary")): 5,
    (("get_mission_info", "today"), ("get_user_history", "monthly_summary")): 5,
    (("submit_mission_result", "success"), ("get_user_history", "weekly_summary")): 5,
    (("submit_mission_result", "fail"), ("get_user_history", "weekly_summary")): 5,
    (("cancel_mission_action", None), ("get_mission_info", "today")): 5,
    (("cancel_mission_action", None), ("get_user_history", "weekly_summary")): 5,
    (("check_mission_equivalency", "behavior"), ("submit_mission_result", "success")): 5,
    (("check_mission_equivalency", "place"), ("submit_mission_result", "success")): 5,
    (("check_mission_equivalency", "behavior"), ("get_mission_info", "today")): 5,
    (("check_mission_equivalency", "place"), ("get_mission_info", "today")): 5,
    (("request_mission_adjustment", "easier"), ("get_mission_info", "today")): 5,
    (("request_mission_adjustment", "harder"), ("get_mission_info", "today")): 5,
}

# Backward-compatible alias for existing single-turn code.
TARGETS = TARGETS_SINGLE

# ============================================================================
# EXTRA HINTS FOR CONTRASTIVE FEW-SHOT
# ============================================================================
# Hint pairs to prevent confusion between similar categories
EXTRA_HINTS = {
    "easier_harder": {
        "description": "Distinguish between 'easier' and 'harder' mission adjustment requests",
        "good_examples": {
            "easier": [
                "이 미션 너무 어려워. 좀 쉽게 해줄래?",
                "미션이 복잡한데, 간단하게 변경해 주실 수 있나요?",
                "이번 미션 난이도를 낮춰 줄 수 있나요?"
            ],
            "harder": [
                "이 미션 너무 쉬워. 좀 더 어렵게 해줄래?",
                "미션의 난이도를 올려서 더 도전적으로 만들어 주세요.",
                "좀 더 어려운 미션을 원해요."
            ]
        }
    },
    "success_fail": {
        "description": "Distinguish between 'success' and 'fail' mission results",
        "good_examples": {
            "success": [
                "미션을 완료했어! 성공했어.",
                "미션 다 했습니다.",
                "미션을 성공적으로 끝냈어요."
            ],
            "fail": [
                "미션을 못 했어. 실패했어.",
                "미션 실패했습니다.",
                "미션을 완료하지 못했어요."
            ]
        }
    },
    "behavior_place": {
        "description": "Distinguish between 'behavior' and 'place' equivalency checks",
        "good_examples": {
            "behavior": [
                "30분 산책하기와 30분 조깅하기가 동등한 미션인가요?",
                "책 읽기와 영상 보기를 같은 미션으로 인정할 수 있나요?",
                "같은 행동으로 간주할 수 있는가?"
            ],
            "place": [
                "공원에서 운동과 집에서 운동이 동등한가요?",
                "카페에서 공부하기와 도서관에서 공부하기는 같나요?",
                "장소가 다르지만 같은 미션인가요?"
            ]
        }
    }
}

# ============================================================================
# DATA SPLIT RATIOS
# ============================================================================
SPLIT_RATIOS = {
    "train": 0.8,
    "valid": 0.1,
    "test": 0.1
}

# ============================================================================
# CSV SCHEMA
# ============================================================================
SEED_CSV_SCHEMA = {
    "user_query": str,
    "function_name": str,
    "arguments": str  # JSON string format
}

GENERATED_CSV_SCHEMA = {
    "user_query": str,
    "function_name": str,
    "arguments": str,
    "generation_round": int
}

GENERATED_MULTI_CSV_SCHEMA = {
    "user_query": str,
    "function_calls": str  # JSON array string format
}

VALIDATED_CSV_SCHEMA = {
    "user_query": str,
    "function_name": str,
    "arguments": str,
    "ambiguous": str  # description of ambiguity if any, empty string if clear
}

VALIDATED_MULTI_CSV_SCHEMA = {
    "user_query": str,
    "function_calls": str,  # JSON array string format
    "ambiguous": str
}

REJECTED_CSV_SCHEMA = {
    "user_query": str,
    "function_name": str,
    "arguments": str,
    "validator_pred": str,  # validator's prediction result (JSON format)
    "reject_stage": str  # "1st_validation" or "2nd_validation"
}

REJECTED_MULTI_CSV_SCHEMA = {
    "user_query": str,
    "function_calls": str,
    "validator_pred": str,
    "reject_stage": str
}

API_ERROR_CSV_SCHEMA = {
    "user_query": str,
    "function_name": str,
    "arguments": str,
    "error_reason": str
}

API_ERROR_MULTI_CSV_SCHEMA = {
    "user_query": str,
    "function_calls": str,
    "error_reason": str
}

# ============================================================================
# VALIDATION RULES
# ============================================================================
VALIDATION_RULES = {
    "warning_rules": [
        {
            "name": "negative_with_success",
            "description": "Negative sentiment (못/안/실패) + success prediction",
            "keywords": ["못", "안 했", "안했", "실패"],
            "rule": "if_any_keyword_and_result_is_success"
        },
        {
            "name": "place_mention_without_place_arg",
            "description": "Place mentioned (집/카페/공원) but not checking place equivalency",
            "keywords": ["집", "카페", "공원", "도서관", "회사", "학교"],
            "rule": "if_keyword_present_and_function_is_not_check_place"
        }
    ]
}

# ============================================================================
# GENERATION PARAMETERS
# ============================================================================
GENERATION_MULTIPLIER = 1.5  # Generate need * 1.5 samples
MAX_GENERATION_ROUNDS = 3  # Maximum rounds to generate data
BATCH_SIZE_PER_ROUND = 10  # Number of samples to generate per round

# ============================================================================
# PATHS
# ============================================================================
DATA_DIR = "data"
OUTPUT_DIR = "output"

SEED_SINGLE_PATH = f"{DATA_DIR}/seed_single.csv"
SEED_MULTI_PATH = f"{DATA_DIR}/seed_multi.csv"

GENERATED_SINGLE_PATH = f"{DATA_DIR}/generated_single.csv"
GENERATED_MULTI_PATH = f"{DATA_DIR}/generated_multi.csv"

VALIDATED_SINGLE_PATH = f"{DATA_DIR}/validated_single.csv"
VALIDATED_MULTI_PATH = f"{DATA_DIR}/validated_multi.csv"

REJECTED_SINGLE_PATH = f"{DATA_DIR}/rejected_single.csv"
REJECTED_MULTI_PATH = f"{DATA_DIR}/rejected_multi.csv"

API_ERROR_SINGLE_PATH = f"{DATA_DIR}/api_error_single.csv"
API_ERROR_MULTI_PATH = f"{DATA_DIR}/api_error_multi.csv"

TRAIN_JSONL_PATH = f"{OUTPUT_DIR}/train.jsonl"
VALID_JSONL_PATH = f"{OUTPUT_DIR}/valid.jsonl"
TEST_JSONL_PATH = f"{OUTPUT_DIR}/test.jsonl"

ANALYSIS_LOG_PATH = f"{OUTPUT_DIR}/analysis.log"
VALIDATION_WARNING_LOG_PATH = f"{OUTPUT_DIR}/validation_warnings.log"
