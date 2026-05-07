# Function Calling Data Augmentation Pipeline

AI-powered data augmentation pipeline specifically designed for fine-tuning Large Language Models on function calling and tool use capabilities.

## 🎯 Purpose

This pipeline generates high-quality training data for teaching LLMs how to:
- Understand function signatures and parameters
- Generate proper function call arguments
- Handle multi-turn conversations with chained function calls
- Validate function call responses

## ✨ Features

- **Single-turn augmentation**: Generate function calling examples for individual API calls
- **Multi-turn augmentation**: Create conversational flows with chained function calls
- **Validation pipeline**: Ensure generated data meets quality standards
- **OpenAI GPT integration**: Leverage advanced language models for data generation
- **JSONL export**: Ready-to-use format for LLM fine-tuning

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full augmentation pipeline
python run.py --mode all

# Generate single-turn function calling data
python run.py --mode single

# Export for fine-tuning
python run.py --mode export
```

## 📊 Data Flow

```
Seed Data → Augmentation → Validation → Export
    ↓           ↓            ↓         ↓
Raw examples → Enhanced → Quality → JSONL
function calls  examples   checked   format
```

## 📁 Project Structure

- `augment.py` - Core augmentation engine with function calling logic
- `validate.py` - Validation pipeline for function call accuracy
- `analyze.py` - Analysis utilities for dataset quality metrics
- `export.py` - Export to fine-tuning ready JSONL format
- `config.py` - Configuration for augmentation parameters
- `run.py` - Main orchestration script

## 🔧 Requirements

- Python 3.8+
- OpenAI API key (set in `.env` file)

## 📝 Usage Examples

### Basic Usage

```bash
python run.py --mode all
```

### Export Only

```bash
python run.py --mode export --format qwen
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details
