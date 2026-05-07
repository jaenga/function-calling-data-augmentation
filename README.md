# Function Calling 데이터 증강 파이프라인

어린이 건강습관 코치 챗봇의 function calling 학습 데이터를 만들기 위한 데이터 증강 및 검증 파이프라인입니다.

Seed 데이터를 기반으로 부족한 intent/function 조합을 분석하고, LLM으로 발화를 증강한 뒤, 2단계 검증을 통과한 데이터만 학습용 JSONL 형식으로 변환합니다.

## 목적

챗봇이 사용자의 자연어 발화를 정확한 함수 호출로 변환할 수 있도록 학습 데이터를 구축합니다.

- 초등학생 말투의 다양한 한국어 발화 생성
- 단일 의도(single-turn)와 복합 의도(multi-turn) 데이터 지원
- function name과 arguments가 올바른지 자동 검증
- 검증 통과 데이터만 fine-tuning용 JSONL로 export

## 전체 흐름

```text
Seed Data
  -> Gap Analysis
  -> LLM 기반 발화 증강
  -> 1차 Function Calling 검증
  -> 2차 Semantic 검증
  -> Qwen JSONL Export
  -> Analysis Report
```

## 주요 기능

- `GapAnalyzer`: function/argument 조합별 부족 데이터 수 계산
- `AugmentationEngine`: OpenAI API를 사용해 한국어 사용자 발화 생성
- `ValidationPipeline`: 생성 데이터를 2단계로 검증
- `ExportPipeline`: 검증 통과 데이터를 Qwen function calling JSONL 형식으로 변환
- `AnalysisEngine`: 통과율, reject stage, 함수/인자 분포, confusion matrix 분석

## 프로젝트 구조

```text
.
├── run.py          # 전체 파이프라인 실행 진입점
├── augment.py      # 데이터 증강
├── validate.py     # 생성 데이터 검증
├── export.py       # JSONL export
├── analyze.py      # 결과 분석 리포트 생성
├── config.py       # 함수 스키마, 목표 분포, 경로 설정
├── data/           # seed/generated/validated/rejected CSV
└── output/         # train/valid/test JSONL 및 분석 로그
```

## 실행 준비

Python 가상환경을 만든 뒤 의존성을 설치합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

프로젝트 루트에 `.env` 파일을 만들고 OpenAI API 키를 설정합니다.

```text
OPENAI_API_KEY=your_api_key_here
```

## 실행 방법

전체 파이프라인 실행:

```bash
python3 run.py --mode all
```

Single-turn 데이터만 증강 및 검증:

```bash
python3 run.py --mode single
```

Multi-turn 데이터만 증강 및 검증:

```bash
python3 run.py --mode multi
```

검증된 데이터를 JSONL로 export:

```bash
python3 run.py --mode export --format qwen
```

## 실행 결과

주요 산출물은 다음 위치에 저장됩니다.

- `data/generated_single.csv`, `data/generated_multi.csv`
- `data/validated_single.csv`, `data/validated_multi.csv`
- `data/rejected_single.csv`, `data/rejected_multi.csv`
- `output/train.jsonl`
- `output/valid.jsonl`
- `output/test.jsonl`
- `output/analysis.log`

## 지원 함수

현재 파이프라인은 챗봇의 미션 관련 기능 호출 데이터를 대상으로 합니다.

- `submit_mission_result`
- `get_mission_info`
- `request_mission_adjustment`
- `check_mission_equivalency`
- `get_user_history`
- `cancel_mission_action`

## 라이선스

MIT License. 자세한 내용은 [LICENSE](LICENSE)를 참고하세요.
