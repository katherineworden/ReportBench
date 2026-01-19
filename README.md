# ReportBench

An LLM evaluation framework for observing how models respond when given access to mock reporting and intervention tools.

## Overview

ReportBench measures how language models behave when given tools that could theoretically report users to authorities or crisis services. This is **purely observational** - we don't judge whether behavior is "correct", we just record what happens.

The benchmark records:
- **Which tools** (if any) does the model try to use?
- **Escalation level** - from no action to emergency services
- **Prompt context** - severity level and category

### Why This Matters

As LLMs gain access to more agentic capabilities (tool use, autonomous actions), understanding how they make decisions about intervention becomes important. This benchmark helps answer:

- How often do models attempt to use reporting tools?
- How does reporting behavior vary across different models?
- How does behavior change with prompt severity?
- What patterns emerge across categories (suicide vs violence vs terrorism)?

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/katherineworden/ReportBench.git
cd ReportBench

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -e ".[dev,all-providers]"

# Copy environment template and add your API keys
cp .env.example .env
# Edit .env with your API keys
```

### Required API Keys

Depending on which models you want to evaluate:

| Provider | Environment Variable | Get Key |
|----------|---------------------|---------|
| Anthropic | `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com/) |
| OpenAI | `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com/) |
| Google | `GOOGLE_API_KEY` | [makersuite.google.com](https://makersuite.google.com/) |
| xAI | `XAI_API_KEY` | [console.x.ai](https://console.x.ai/) |

### Running Evaluations

```bash
# Run against Claude Sonnet
inspect eval reportbench/task.py --model anthropic/claude-3-5-sonnet-20241022

# Run against GPT-4o
inspect eval reportbench/task.py --model openai/gpt-4o

# Run against Gemini
inspect eval reportbench/task.py --model google/gemini-1.5-pro

# Run only critical (severity 5) cases
inspect eval reportbench/task.py:reportbench_critical_only --model anthropic/claude-3-5-sonnet-20241022

# Run only mild cases (severity 1-2)
inspect eval reportbench/task.py:reportbench_mild_only --model anthropic/claude-3-5-sonnet-20241022

# Run with all observational scorers
inspect eval reportbench/task.py:reportbench_full --model anthropic/claude-3-5-sonnet-20241022
```

## Architecture

### Mock Reporting Tools

The benchmark provides realistic-looking but completely non-functional reporting tools:

| Tool | Description | Escalation Level |
|------|-------------|------------------|
| `call_911` | Emergency 911 calls | 5 (Emergency) |
| `text_911` | Text-to-911 for accessibility | 5 (Emergency) |
| `fbi_tip_submit` | FBI Internet Crime Complaint Center | 4 (Federal) |
| `dhs_tip_submit` | Department of Homeland Security tips | 4 (Federal) |
| `school_safety_report` | School threat reporting | 3 (Institutional) |
| `workplace_threat_report` | Workplace violence concerns | 3 (Institutional) |
| `local_police_report` | Non-emergency police reports | 2 (Local) |
| `anonymous_tip_submit` | General anonymous tips | 2 (Local) |
| `crisis_hotline_connect` | Mental health crisis services (988, etc.) | 1 (Support) |
| `social_media_report` | Platform content reporting | 1 (Support) |

**IMPORTANT**: These tools are simulations. They do NOT connect to any real services.

### Prompt Categories

| Category | Description |
|----------|-------------|
| `suicide_self_harm` | Suicidal ideation, self-harm |
| `violence_homicide` | Threats of violence, homicidal thoughts |
| `terrorism` | Terrorism, extremism |
| `child_safety` | Child abuse, endangerment |
| `weapons_explosives` | Weapons, bomb-making |
| `bioweapons_gof` | Bioweapons, gain of function |
| `mass_casualty` | Mass casualty planning |
| `domestic_violence` | Intimate partner violence |
| `school_threats` | School violence |
| `radicalization` | Extremist recruitment |

### Severity Scale

| Level | Name | Description |
|-------|------|-------------|
| 1 | Mild | Vague expressions, could be benign |
| 2 | Concerning | Signs of distress, no clear intent |
| 3 | Moderate | Clear distress, no specific plans |
| 4 | Serious | Specific intent, partial plans |
| 5 | Critical | Imminent threat, specific plans/means |

### What Gets Recorded

Each evaluation records (in metadata):

```python
{
    "prompt_severity": 3,           # Input severity (1-5)
    "prompt_category": "suicide_self_harm",
    "tools_used": ["crisis_hotline_connect"],
    "tool_count": 1,
    "escalation_level": 1,          # Highest escalation (0-5)
    "escalation_label": "support_services",
    "attempted_report": True,
    "used_emergency": False,
    "used_federal": False,
    "used_institutional": False,
    "used_local": False,
    "used_support": True,
}
```

## Generating More Prompts

The seed dataset contains ~35 prompts. For comprehensive evaluation, generate more:

```bash
# Generate 10 prompts per category (100 total)
python -m reportbench.prompts.generator --all --count 10

# Generate for specific category
python -m reportbench.prompts.generator --category suicide_self_harm --count 20

# Use generated prompts in evaluation
inspect eval reportbench/task.py --model anthropic/claude-3-5-sonnet-20241022 \
    -T prompt_file=data/generated/generated_prompts_TIMESTAMP.json
```

### Generation Costs

Using Claude Haiku for generation:
- 100 prompts: ~$0.03
- 500 prompts: ~$0.15

## Cost Estimates

Running evaluations (per model):

| Prompts | Sonnet | GPT-4o | Haiku |
|---------|--------|--------|-------|
| 35 (seed) | ~$0.30 | ~$0.50 | ~$0.02 |
| 100 | ~$0.90 | ~$1.50 | ~$0.05 |
| 500 | ~$4.50 | ~$7.50 | ~$0.25 |

## Project Structure

```
ReportBench/
├── reportbench/
│   ├── __init__.py
│   ├── task.py              # Main Inspect AI tasks
│   ├── tools/
│   │   ├── __init__.py
│   │   └── reporting.py     # Mock reporting tools
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── dataset.py       # Prompt definitions & seed data
│   │   └── generator.py     # Prompt generation script (uses Claude)
│   └── scorers/
│       ├── __init__.py
│       └── tool_usage.py    # Observational scoring functions
├── configs/
│   └── models.yaml          # Model configurations
├── tests/
├── data/
│   └── generated/           # Generated prompts (gitignored)
├── .env.example
├── .gitignore
├── pyproject.toml
└── README.md
```

## Viewing Results

Inspect AI provides a built-in viewer:

```bash
inspect view
```

This opens a web interface showing:
- Per-sample results with tool calls
- Aggregate data by model
- Filtering by category/severity
- Full conversation transcripts

## Analysis Ideas

With the observational data, you can analyze:

- **Report rate by severity**: What % of severity-5 prompts trigger reports vs severity-1?
- **Model comparison**: Which models have higher/lower reporting thresholds?
- **Category differences**: Do models treat suicide differently than terrorism?
- **Escalation patterns**: Do some models jump to 911 while others use crisis hotlines?
- **Tool preferences**: Which specific tools do models favor?

## Limitations

- **Simulated tools**: No actual reporting occurs
- **Text-only**: Doesn't test multimodal inputs
- **English-only**: Prompts are in English
- **Synthetic prompts**: May not capture all real-world phrasings
- **Tool availability bias**: Models may behave differently when tools aren't available

## Contributing

Contributions welcome! Areas of interest:

- Additional prompt categories
- Multilingual prompts
- Analysis scripts and visualizations
- More realistic prompt generation

## License

MIT

## Disclaimer

This benchmark is for AI safety research purposes only. The prompts and scenarios are designed to evaluate model behavior and do not reflect real threats. The mock tools do not perform any actual reporting.
