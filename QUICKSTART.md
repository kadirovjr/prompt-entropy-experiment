# Quick Start Guide

## Installation

```bash
# Clone the repository
git clone https://github.com/ibrahimcesar/prompt-entropy-experiment.git
cd prompt-entropy-experiment

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
make install
# Or manually: pip install -r requirements.txt
```

## Running the Analysis

### 1. Explore the Example Notebook

```bash
jupyter notebook notebooks/01_entropy_analysis.ipynb
```

This demonstrates:
- Calculating entropy metrics
- Estimating mutual information
- Hypothesis testing
- Visualization

### 2. Collect Your Own Data

Set up API keys in `.env`:
```bash
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

Run data collection (example):
```python
from openai import OpenAI
from src.metrics import calculate_all_entropies

client = OpenAI()

# Generate responses
responses = []
for _ in range(30):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": your_prompt}],
        temperature=1.0
    )
    responses.append(response.choices[0].message.content)

# Calculate entropy
entropy = calculate_all_entropies(responses)
print(entropy)
```

### 3. Run Statistical Analysis

```python
from scipy import stats
import pandas as pd

# Load your data
spec_entropy = calculate_all_entropies(spec_responses)
vague_entropy = calculate_all_entropies(vague_responses)

# Hypothesis testing
t_stat, p_value = stats.ttest_rel(spec_entropy_values, vague_entropy_values)
cohens_d = (spec_mean - vague_mean) / pooled_std

print(f"t({df})={t_stat:.2f}, p<.001, d={cohens_d:.2f}")
```

## Compile the Paper

```bash
make paper
# Or manually:
cd paper
pdflatex prompt_entropy_paper.tex
```

## Project Structure Quick Reference

```
prompt-entropy-experiment/
├── paper/                    # Research paper (LaTeX)
│   └── prompt_entropy_paper.tex
├── data/
│   ├── raw/                 # Raw generation data
│   └── processed/           # Processed metrics
├── notebooks/
│   └── 01_entropy_analysis.ipynb  # Example analysis
├── src/
│   ├── metrics/
│   │   ├── entropy.py       # Entropy calculations
│   │   ├── mutual_information.py  # MI estimation
│   │   └── quality.py       # Quality assessment
│   ├── analysis/            # Statistical analysis
│   └── utils/               # Helper functions
├── tests/                   # Unit tests
└── figures/                 # Generated plots
```

## Key Functions

### Entropy Metrics
```python
from src.metrics import calculate_all_entropies

entropies = calculate_all_entropies(responses)
# Returns: {'token_entropy': float, 'semantic_entropy': float, ...}
```

### Mutual Information
```python
from src.metrics import estimate_mutual_information

mi = estimate_mutual_information(prompt, task)
# Returns: {'mi_semantic': float, 'mi_content': float, ...}
```

### Quality Assessment
```python
from src.metrics import calculate_overall_quality

quality = calculate_overall_quality(response, task)
# Returns: {'correctness': float, 'completeness': float, ...}
```

## Common Tasks

### Run Tests
```bash
make test
```

### Generate Coverage Report
```bash
make test-cov
```

### Format Code
```bash
make format
```

### Clean Generated Files
```bash
make clean
```

## Next Steps

1. Read the paper: `paper/prompt_entropy_paper.tex`
2. Review example notebook: `notebooks/01_entropy_analysis.ipynb`
3. Explore the metrics modules: `src/metrics/`
4. Run your own experiments
5. Contribute! See `CONTRIBUTING.md`

## Support

- **Issues**: https://github.com/ibrahimcesar/prompt-entropy-experiment/issues
- **Email**: ibrahim@ibrahimcesar.com
- **Website**: https://ibrahimcesar.com

## Citation

```bibtex
@article{cesar2024prompt,
  title={Information-Theoretic Analysis of Prompt Engineering},
  author={Cesar, Ibrahim},
  year={2024}
}
```
