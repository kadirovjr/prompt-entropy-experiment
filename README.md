<div align="right">
  
**"The limits of my language mean the limits of my world."**  
  — Ludwig Wittgenstein, _Philosophical Investigations_
  
</div>

<img src="abacus_1f9ee.webp" alt="Abacus" title="Abacus Emoji from Telegram (2023)" height="150" />

# Prompt Entropy Experiment

Empirical validation of information-theoretic principles in prompt engineering for generative AI systems.

## Research Question

Can we quantify prompt quality using Shannon entropy and mutual information? Does the effect persist across different sampling regimes (temperature settings)?

## Hypotheses

**H1 (Primary):** Specification-driven prompts reduce output entropy across all temperatures
**H2:** Entropy increases monotonically with temperature (validation)
**H3:** Interaction effect - Does the entropy difference persist, amplify, or converge with temperature?
**H4:** Mutual information correlates negatively with entropy across all temperatures

## Experimental Design

**Multi-temperature study** across:
- **30 tasks** spanning 6 domains
- **2 prompt types** (specification-driven vs. vague)
- **2 models** (GPT-4, Claude-3.5 Sonnet)
- **3 temperatures** (0.7 production, 1.0 baseline, 1.2 exploration)
- **30 samples** per condition

**Total planned generations:** 10,800 (30 tasks × 2 prompts × 2 models × 3 temps × 30 samples)

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

## Temperature Framework

The study uses **three strategic temperatures** to validate findings across sampling regimes:

- **T=0.7 (Production)**: Real-world deployment setting, practical relevance
- **T=1.0 (Baseline)**: Natural, unscaled probability distribution, theoretical purity
- **T=1.2 (Exploration)**: Latent space exploration, robustness validation

This design tests whether the entropy-reducing effect of specification prompts is:
1. **Production-valid**: Does it hold in real-world settings? (0.7)
2. **Theoretically sound**: What is the effect at the pure distribution? (1.0)
3. **Robust**: Does it persist during latent exploration? (1.2)

## Methodology

### Entropy Metrics
- **Token Entropy**: Shannon entropy over token distributions
- **Semantic Entropy**: Clustering-based entropy in embedding space
- **Structural Entropy**: Entropy over structural features

### Mutual Information Estimation
- **MI Content**: Information content indicators (numbers, constraints, examples)
- **MI Coverage**: Task concept coverage
- **MI Semantic**: Embedding similarity
- **MI Combined**: Weighted combination

### Quality Evaluation
- Correctness (35%)
- Completeness (25%)
- Relevance (20%)
- Coherence (10%)
- Format Compliance (10%)

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/ibrahimcesar/prompt-entropy-experiment.git
cd prompt-entropy-experiment
make setup

# 2. Configure API keys
make init-env  # Creates .env template
# Edit .env with your API keys

# 3. Run experiment
make run-experiment EXPERIMENT=exp001 CONFIG=config/tasks.json

# Or run multi-temperature study (recommended)
make run-temperature-study EXPERIMENT=temp_comprehensive
```

## Usage

### Command-Line (Recommended)

```bash
# Full 3-temperature study (production/baseline/exploration)
make run-temperature-study EXPERIMENT=temp001

# Single temperature baseline
make run-experiment EXPERIMENT=baseline TEMPERATURE=1.0

# Quick pilot study (3 tasks, 5 samples per condition)
make run-temperature-study-small EXPERIMENT=pilot
```

### Python API

```python
from src.metrics import calculate_entropy, estimate_mutual_information
from src.sampling import sample_responses

# Sample responses from LLM
responses = sample_responses(
    prompt=prompt,
    model="gpt-4",
    n=30,
    temperature=1.0
)

# Calculate entropy
entropy = calculate_entropy(responses, metric="token")

# Estimate mutual information
mi = estimate_mutual_information(prompt, task)
```

All experiments include **full audit logging** with git state, parameters, file hashes, and timestamps for reproducibility.

## Documentation

**Comprehensive methodology documentation:**
- **[METHODOLOGY.md](METHODOLOGY.md)**: Complete experimental protocol with formal hypotheses, temperature framework, and reproducibility guidelines
- **[QUICKSTART.md](QUICKSTART.md)**: Quick reference guide
- **[CONTRIBUTING.md](CONTRIBUTING.md)**: Contribution guidelines

**Academic paper:**
- **LaTeX source**: `paper/prompt_entropy_paper.tex`
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
@article{cesar2025prompt,
  title={Information-Theoretic Analysis of Prompt Engineering:
         Multi-Temperature Validation of Entropy and Mutual Information Effects},
  author={Cesar, Ibrahim},
  journal={arXiv preprint},
  year={2025}
}
```

## Project Status

**Data Collection Phase**: Implementing multi-temperature experimental design to validate the robustness of entropy-based prompt quality metrics across different sampling regimes.

**Next Steps:**
1. Complete 3-temperature data collection (T=0.7, 1.0, 1.2)
2. Statistical analysis of main effects and interactions
3. Publication preparation

This study serves as foundational research before deeper integration into the Categorical Operations Management framework.

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

Thanks to the broader AI research community for developing the theoretical foundations this work builds upon, and to Anthropic and OpenAI for making Claude-3.5 Sonnet and GPT-4 available for research.
