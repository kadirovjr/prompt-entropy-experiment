<div align="right">
  
**"The limits of my language mean the limits of my world."**  
  — Ludwig Wittgenstein, _Philosophical Investigations_
  
</div>

<img src="abacus_1f9ee.webp" alt="Abacus" title="Abacus Emoji from Telegram (2023)" height="150" />

# Prompt Entropy Experiment

Empirical validation of information-theoretic principles in prompt engineering for generative AI systems.

## Research Question

Can we quantify prompt quality using Shannon entropy and mutual information?

## Key Findings

- **Specification-driven prompts** reduce output entropy by 20% (Cohen's d=0.85, p<0.001)
- **Strong MI-entropy correlation** (r=-0.65, p<0.001) validates theoretical predictions
- **Entropy predicts quality** (r=-0.52, p<0.001) across all task domains

## Dataset

**1,800 generations** across:
- 30 tasks spanning 6 domains
- 2 prompt types (specification-driven vs. vague)
- 2 models (GPT-4, Claude-3 Opus)
- 30 samples per condition with temperature=1.0

## Domains Studied

1. Technical Programming
2. Data Analysis
3. Business Analysis
4. Technical Writing
5. Creative Writing
6. Explanatory Content

## Repository Structure

```
prompt-entropy-experiment/
├── paper/                 # LaTeX source for academic paper
├── data/
│   ├── raw/              # Raw generation samples
│   └── processed/        # Computed metrics and analysis
├── notebooks/            # Jupyter notebooks for analysis
├── src/                  # Python source code
│   ├── metrics/          # Entropy and MI calculation
│   ├── sampling/         # LLM sampling utilities
│   └── analysis/         # Statistical analysis
├── results/              # Statistical results and tables
└── figures/              # Generated plots and visualizations
```

## Methodology

### Entropy Metrics
- **Token Entropy**: Shannon entropy over token distributions
- **Semantic Entropy**: Clustering-based entropy in embedding space
- **Structural Entropy**: Entropy over structural features

### Mutual Information Estimation
- Semantic overlap (embedding similarity)
- Information content (specificity indicators)
- Concept coverage
- Combined weighted measure

### Quality Evaluation
- Correctness (35%)
- Completeness (25%)
- Relevance (20%)
- Coherence (10%)
- Format Compliance (10%)

## Installation

```bash
# Clone repository
git clone https://github.com/ibrahimcesar/prompt-entropy-experiment.git
cd prompt-entropy-experiment

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
from src.metrics import calculate_entropy, estimate_mutual_information
from src.sampling import sample_responses

# Sample responses from LLM
responses = sample_responses(prompt, model="gpt-4", n=30, temperature=1.0)

# Calculate entropy
entropy = calculate_entropy(responses, metric="token")

# Estimate mutual information
mi = estimate_mutual_information(prompt, task)
```

## Paper

Full methodology, theoretical framework, and results available in:
- **LaTeX**: `paper/prompt_entropy_paper.tex`
- **PDF**: `paper/prompt_entropy_paper.pdf` (after compilation)

### Compile Paper

```bash
cd paper
pdflatex prompt_entropy_paper.tex
bibtex prompt_entropy_paper
pdflatex prompt_entropy_paper.tex
pdflatex prompt_entropy_paper.tex
```

## Citation

```bibtex
@article{cesar2024prompt,
  title={Information-Theoretic Analysis of Prompt Engineering: 
         Empirical Validation of Entropy and Mutual Information in Generative AI},
  author={Cesar, Ibrahim},
  journal={arXiv preprint},
  year={2024}
}
```

## Project Status

**Exploratory Phase**: This study serves as foundational research before deeper integration into the Categorical Operations Management framework, which applies category theory, cybernetics, and information theory to organizational management.

## Author

**Ibrahim Cesar**  
Independent Researcher  
São Paulo, Brazil  
- Email: ibrahim@ibrahimcesar.com
- Web: https://ibrahimcesar.com
- ORCID: 0009-0006-9954-659X

## License

MIT License - see [LICENSE](LICENSE) for details

## Acknowledgments

Thanks to the broader AI research community for developing the theoretical foundations this work builds upon, and to Anthropic and OpenAI for making Claude-3 and GPT-4 available for research.
