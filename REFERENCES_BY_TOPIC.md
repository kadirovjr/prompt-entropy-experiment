# References Organized by Research Topics

Quick reference guide mapping papers to specific research questions and hypotheses.

---

## For H1: Specification-Driven Prompts Reduce Output Entropy

### Theoretical Foundation:
- **Shannon (1948)** - Entropy definition H(X) = -Σ p(x) log p(x)
- **Cover & Thomas (2006)** - Data processing inequality, conditioning reduces entropy
- **Polyanskiy & Wu (2019)** - Fano's inequality for information bounds

### Empirical Support:
- **Kuhn et al. (2023, 2024)** - Semantic entropy in LLMs, shows prompting affects uncertainty
- **Zhang et al. (2018)** - Entropy characterizes text diversity
- **White et al. (2023)** - Prompt patterns that increase specificity

### Prompt Engineering Context:
- **Wei et al. (2022)** - Chain-of-thought increases specification → reduces entropy
- **Zhou et al. (2023)** - Automated prompt optimization reduces uncertainty
- **Sahoo et al. (2024)** - Survey of specificity-increasing techniques

---

## For H2: Entropy Increases Monotonically with Temperature

### Core Theory:
- **Shannon (1948)** - Temperature scaling affects probability distribution
- **Cover & Thomas (2006)** - Entropy is concave in distribution

### Sampling Strategies:
- **Holtzman et al. (2019)** - Nucleus sampling, temperature effects on distribution
- **Robinson et al. (2024)** - Empirical study: temperature 0.0-2.0, task-dependent effects
- **Zhang et al. (2025)** - Multi-temperature optimization strategies
- **Fan et al. (2018)** - Temperature in creative generation

### Implementation:
- **Brown et al. (2020)** - GPT-3 temperature parameter
- **Achiam et al. (2023)** - GPT-4 temperature range 0.0-2.0
- **Anthropic (2024)** - Claude temperature implementation

---

## For H3: Interaction Effect - Temperature × Prompt Type

### Theoretical Predictions:
- **Cover & Thomas (2006)** - Data processing inequality: I(X;Y|T) relationship
- **Kullback & Leibler (1951)** - KL divergence and distribution distance

### Empirical Studies:
- **Robinson et al. (2024)** - Temperature effects vary by task complexity
- **Kuhn et al. (2023)** - Semantic entropy across different settings
- **Farquhar et al. (2024)** - Beyond semantic entropy: cluster analysis

### Related Work:
- **Zhang et al. (2024)** - Diversity preservation across temperatures
- **Lin et al. (2024)** - Kernel language entropy generalizes across settings

---

## For H4: Mutual Information Correlates with Quality

### Information Theory:
- **Shannon (1948)** - Mutual information I(X;Y) = H(Y) - H(Y|X)
- **Cover & Thomas (2006)** - MI properties, data processing inequality
- **Kullback & Leibler (1951)** - KL divergence as MI component

### Quality and Uncertainty:
- **Kuhn et al. (2024)** - Semantic entropy predicts hallucinations (Nature)
- **Guan et al. (2024)** - UQ survey: entropy-quality relationship
- **Malinin & Gales (2018)** - Predictive uncertainty in neural networks

### Evaluation Metrics:
- **Liu et al. (2023)** - G-EVAL: LLM-based quality assessment
- **Zhao et al. (2023)** - DiscoScore: coherence evaluation
- **Zheng et al. (2024)** - MT-Bench: multi-dimensional quality

---

## For Semantic Entropy Metric

### Core Papers:
- **Kuhn et al. (2023)** - Original semantic entropy paper (arXiv)
- **Kuhn et al. (2024)** - Nature paper: hallucination detection
- **Farquhar et al. (2024)** - Beyond semantic entropy: pairwise similarity
- **Lin et al. (2024)** - Kernel language entropy (NeurIPS)

### Clustering Methods:
- **Devlin et al. (2019)** - BERT for NLI-based clustering
- **Reimers & Gurevych (2019)** - Sentence-BERT for embeddings
- **Gao et al. (2021)** - SimCSE: contrastive embeddings

### Embedding Models:
- **Neelakantan et al. (2022)** - text-embedding-ada-002
- **OpenAI (2024)** - text-embedding-3-small/large
- **Muennighoff et al. (2023)** - MTEB benchmark

---

## For Mutual Information Estimation

### Theoretical Background:
- **Shannon (1948)** - MI definition and properties
- **Cover & Thomas (2006)** - MI estimation methods
- **Polyanskiy & Wu (2019)** - Information bounds

### Semantic Similarity:
- **Reimers & Gurevych (2019)** - Sentence-BERT cosine similarity
- **Gao et al. (2021)** - SimCSE embeddings
- **Dar et al. (2023)** - Analyzing transformers in embedding space

### Content Analysis:
- **White et al. (2023)** - Prompt pattern analysis
- **Sahoo et al. (2024)** - Prompt engineering survey
- **Zhou et al. (2023)** - Automated prompt optimization

---

## For Quality Metrics

### Traditional Metrics:
- **Papineni et al. (2002)** - BLEU: n-gram precision
- **Lin (2004)** - ROUGE: recall-oriented metrics
- **Banerjee & Lavie (2005)** - METEOR: semantic matching

### Modern Approaches:
- **Zhao et al. (2023)** - DiscoScore: coherence with BERT
- **Liu et al. (2023)** - G-EVAL: GPT-4 based evaluation
- **Zheng et al. (2024)** - MT-Bench: LLM-as-judge

### Quality Dimensions:
- **Kuhn et al. (2024)** - Correctness via hallucination detection
- **Chen et al. (2024)** - Comprehensive UQ benchmark
- **Li et al. (2025)** - Consistency evaluation

---

## For Multi-Temperature Study Design

### Temperature Effects:
- **Robinson et al. (2024)** - Systematic temperature study 0.0-2.0
- **Zhang et al. (2025)** - Multi-sample temperature optimization
- **Holtzman et al. (2019)** - Sampling strategy comparison

### Production Settings:
- **Brown et al. (2020)** - GPT-3 default temp=0.7
- **Achiam et al. (2023)** - GPT-4 temp=1.0 default
- **Anthropic (2024)** - Claude temperature recommendations

### Statistical Validation:
- **Cover & Thomas (2006)** - Statistical properties of entropy
- **Polyanskiy & Wu (2019)** - Estimation bounds
- **Chen et al. (2024)** - UQ method benchmarking

---

## For Domain Analysis (Technical/Creative/Business)

### Technical Domains:
- **Wei et al. (2022)** - CoT for reasoning tasks
- **Brown et al. (2020)** - Few-shot for code generation
- **Robinson et al. (2024)** - Problem-solving tasks

### Creative Domains:
- **Fan et al. (2018)** - Hierarchical story generation
- **Zhang et al. (2018)** - Diversity in generation
- **Holtzman et al. (2019)** - Avoiding degeneration

### Business/Analysis:
- **White et al. (2023)** - Prompt patterns for different domains
- **Sahoo et al. (2024)** - Domain-specific prompting
- **Gao et al. (2024)** - RAG for knowledge-intensive tasks

---

## For Model Comparison (GPT-4 vs Claude)

### GPT-4:
- **Achiam et al. (2023)** - GPT-4 technical report
- **Brown et al. (2020)** - GPT-3 foundation
- **Radford et al. (2019)** - GPT-2 architecture

### Claude:
- **Anthropic (2024)** - Claude 3 family
- **Bai et al. (2022)** - Constitutional AI
- **Lambert et al. (2025)** - RLHF book (covers both models)

### Comparison Studies:
- **Zheng et al. (2024)** - MT-Bench cross-model comparison
- **Liu et al. (2023)** - G-EVAL on multiple models
- **Chen et al. (2024)** - UQ across different LLMs

---

## For Prompt Specification Techniques

### Few-Shot Learning:
- **Brown et al. (2020)** - GPT-3 few-shot paradigm
- **Kojima et al. (2022)** - Zero-shot reasoning

### Chain-of-Thought:
- **Wei et al. (2022)** - Original CoT paper
- **Wang et al. (2022)** - Self-consistency
- **Yao et al. (2023)** - Tree of thoughts

### Structured Prompts:
- **White et al. (2023)** - Prompt pattern catalog
- **Reynolds & McDonell (2021)** - Prompt programming
- **Zhou et al. (2023)** - Automated optimization

### RAG Integration:
- **Gao et al. (2024)** - RAG survey
- **Ji et al. (2025)** - Emulating RAG via prompting

---

## For Statistical Analysis Methods

### Hypothesis Testing:
- **Cover & Thomas (2006)** - Information-theoretic tests
- **Polyanskiy & Wu (2019)** - Statistical bounds
- **Robinson et al. (2024)** - Temperature effect analysis

### Effect Sizes:
- **Kuhn et al. (2024)** - Effect sizes in entropy studies
- **Wang et al. (2022)** - Self-consistency improvements
- **Wei et al. (2022)** - CoT effect sizes

### Correlation Analysis:
- **Kuhn et al. (2023)** - Entropy-uncertainty correlation
- **Chen et al. (2024)** - UQ method correlations
- **Zhao et al. (2023)** - Coherence-quality correlation

---

## For Limitations and Future Work

### Entropy Estimation:
- **Polyanskiy & Wu (2019)** - Estimation bounds and limitations
- **Kuhn et al. (2023)** - Semantic entropy limitations
- **Chen et al. (2024)** - UQ method comparison

### MI Estimation:
- **Cover & Thomas (2006)** - MI estimation challenges
- **Reimers & Gurevych (2019)** - Embedding similarity as MI proxy

### Quality Metrics:
- **Papineni et al. (2002)** - BLEU limitations
- **Liu et al. (2023)** - Need for better metrics
- **Zheng et al. (2024)** - Evaluation challenges

### Model Coverage:
- **Achiam et al. (2023)** - GPT-4 capabilities and limits
- **Anthropic (2024)** - Claude model family
- **Sahoo et al. (2024)** - Generalization across models

---

## For Safety and Robustness

### Adversarial Attacks:
- **Zou et al. (2023)** - Jailbreaking aligned models
- **Perez & Ribeiro (2022)** - Prompt injection

### Defense Mechanisms:
- **Robey et al. (2023)** - SmoothLLM defense
- **Chao et al. (2024)** - JailbreakBench
- **Bai et al. (2022)** - Constitutional AI

---

## For Training and Alignment

### RLHF:
- **Ouyang et al. (2022)** - InstructGPT
- **Lambert et al. (2025)** - RLHF book
- **Rafailov et al. (2024)** - DPO alternative

### Fine-Tuning:
- **Zhang et al. (2024)** - Preserving diversity
- **Teng et al. (2024)** - Multi-turn dialogue
- **Bai et al. (2022)** - Constitutional AI

---

## Quick Lookup by Year

### Foundational (Pre-2018):
- Shannon (1948), Kullback & Leibler (1951), Fano (1961), Jelinek (1977), Papineni (2002), Lin (2004), Banerjee (2005), Cover & Thomas (2006)

### Early Deep Learning (2017-2019):
- Vaswani (2017), Malinin (2018), Zhang (2018), Fan (2018), Devlin (2019), Holtzman (2019), Radford (2019), Reimers (2019), Polyanskiy (2019)

### GPT-3 Era (2020-2022):
- Brown (2020), Reynolds (2021), Gao (2021), Wei (2022), Wang (2022), Kojima (2022), Neelakantan (2022), Bai (2022), Ouyang (2022)

### ChatGPT Era (2023):
- Achiam (2023), White (2023), Kuhn (2023), Zou (2023), Zhou (2023), Zhao (2023), Liu (2023), Yao (2023), Robey (2023), Muennighoff (2023), Dar (2023), Carpenter (2023)

### Recent Advances (2024):
- Anthropic (2024), OpenAI (2024), Kuhn (2024 Nature), Robinson (2024), Farquhar (2024), Lin (2024), Rafailov (2024), Zheng (2024), Gao (2024), Sahoo (2024), Chao (2024), Chen (2024), Zhang (2024), Teng (2024), Guan (2024), Ouyang (2024)

### Cutting Edge (2025):
- Lambert (2025), Zhang (2025), Li (2025), Ji (2025)

---

## Citation Recommendations for Paper Sections

### Abstract:
- Shannon (1948), Cover & Thomas (2006), Kuhn et al. (2024)

### Introduction:
- Brown (2020), White (2023), Sahoo (2024)

### Background - Information Theory:
- Shannon (1948), Cover & Thomas (2006), Kullback & Leibler (1951)

### Background - Prompt Engineering:
- Wei (2022), White (2023), Brown (2020)

### Background - Entropy in LLMs:
- Kuhn (2023, 2024), Jelinek (1977), Malinin (2018)

### Methodology - Temperature:
- Holtzman (2019), Robinson (2024), Zhang (2025)

### Methodology - Metrics:
- Shannon (1948), Kuhn (2023), Reimers (2019)

### Results - H1:
- Cover & Thomas (2006), Kuhn (2024), Wei (2022)

### Results - H2:
- Robinson (2024), Holtzman (2019), Fan (2018)

### Results - H3:
- Cover & Thomas (2006), Kuhn (2023), Farquhar (2024)

### Results - H4:
- Shannon (1948), Kuhn (2024), Liu (2023)

### Discussion - Implications:
- Cover & Thomas (2006), White (2023), Sahoo (2024)

### Discussion - Limitations:
- Polyanskiy (2019), Chen (2024), Kuhn (2023)

### Future Work:
- Lin (2024), Farquhar (2024), Lambert (2025)

---

**Total References by Category:**
- Information Theory: 5
- Entropy in LLMs: 8
- Prompt Engineering: 5
- Sampling/Temperature: 4
- Foundation Models: 6
- Advanced Prompting: 4
- Evaluation: 6
- Embeddings: 6
- Alignment: 3
- Safety: 4
- Training: 2
- Additional: 4

**Total: 75+ papers/books**

**Coverage:**
- Foundational theory: ✓
- Recent empirical work (2023-2025): ✓
- Both GPT and Claude ecosystems: ✓
- Multiple evaluation approaches: ✓
- Safety and robustness: ✓
- Diverse application domains: ✓
