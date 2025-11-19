# Recommended Reading List
## Prioritized for Information-Theoretic Analysis of Prompt Engineering

**Purpose:** Essential papers for understanding and extending this research, prioritized by relevance and importance.

---

## TIER 1: Absolutely Essential (Must Read)

### Information Theory Core
1. **Shannon, C. E. (1948). A Mathematical Theory of Communication**
   - WHY: Defines entropy H(X) and mutual information I(X;Y) - the mathematical foundation of this entire research
   - READ: Sections 1-6 (skip Section 7 on continuous case for now)
   - TIME: 3-4 hours
   - TAKEAWAY: Understanding why H(X) measures uncertainty and I(X;Y) measures shared information

2. **Cover, T. M., & Thomas, J. A. (2006). Elements of Information Theory**
   - WHY: Modern comprehensive reference, explains data processing inequality and Fano's inequality
   - READ: Chapter 2 (Entropy and Mutual Information), Chapter 8 (Differential Entropy)
   - TIME: 6-8 hours
   - TAKEAWAY: Mathematical tools for theoretical predictions H1-H4

### Semantic Entropy (Core to Our Method)
3. **Kuhn, L., et al. (2023). Semantic Uncertainty**
   - WHY: Introduces the exact semantic entropy method we use - clustering in embedding space
   - READ: Entire paper (12 pages)
   - TIME: 2-3 hours + implementation review
   - TAKEAWAY: How to compute entropy over meanings rather than token sequences

4. **Kuhn, L., et al. (2024). Detecting Hallucinations Using Semantic Entropy (Nature)**
   - WHY: Validates that semantic entropy predicts quality - supports H3 and H4
   - READ: Full paper
   - TIME: 1-2 hours
   - TAKEAWAY: Empirical evidence that entropy-based metrics work for LLM evaluation

### Temperature and Sampling
5. **Robinson, M., et al. (2024). Effect of Sampling Temperature on Problem Solving**
   - WHY: Most recent comprehensive study on temperature effects (0.0-2.0), directly relevant to H2 and H3
   - READ: Full paper
   - TIME: 2 hours
   - TAKEAWAY: Temperature effects are task-dependent, justifies multi-temperature validation

6. **Holtzman, A., et al. (2019). The Curious Case of Neural Text Degeneration**
   - WHY: Introduces nucleus sampling, explains temperature vs top-p, essential for understanding sampling
   - READ: Sections 1-4
   - TIME: 1-2 hours
   - TAKEAWAY: Why pure temperature sampling can fail, sampling strategy implications

### Prompt Engineering
7. **Wei, J., et al. (2022). Chain-of-Thought Prompting**
   - WHY: Shows how prompt structure (CoT) dramatically affects performance - evidence for specification importance
   - READ: Full paper
   - TIME: 1-2 hours
   - TAKEAWAY: Intermediate steps = specification → better results

8. **White, J., et al. (2023). Prompt Pattern Catalog**
   - WHY: Practical catalog of what makes prompts specific vs vague - operationalizes our prompt design
   - READ: Sections 1-3, skim pattern catalog
   - TIME: 2-3 hours
   - TAKEAWAY: Concrete patterns for specification-driven prompts

---

## TIER 2: Highly Recommended (Important Context)

### Foundation Models
9. **Brown, T., et al. (2020). Language Models are Few-Shot Learners (GPT-3)**
   - WHY: Establishes few-shot paradigm, shows why prompting matters
   - READ: Abstract, Introduction, Sections 2-3
   - TIME: 2-3 hours
   - TAKEAWAY: Scale + prompting = capability

10. **Vaswani, A., et al. (2017). Attention Is All You Need**
    - WHY: Transformer architecture underlies GPT-4 and Claude
    - READ: Sections 1-3 (architecture), skim Section 4 (training)
    - TIME: 2-3 hours
    - TAKEAWAY: How attention enables contextualized representations

11. **Achiam, J., et al. (2023). GPT-4 Technical Report**
    - WHY: One of the two models used in this research
    - READ: Sections 1-4, skim evaluation sections
    - TIME: 2 hours
    - TAKEAWAY: GPT-4 capabilities and limitations

### Advanced Entropy Methods
12. **Farquhar, S., et al. (2024). Beyond Semantic Entropy**
    - WHY: Extends semantic entropy with pairwise similarity - potential improvement for our metrics
    - READ: Full paper
    - TIME: 1-2 hours
    - TAKEAWAY: Intra-cluster and inter-cluster similarity matter

13. **Lin, A., et al. (2024). Kernel Language Entropy (NeurIPS)**
    - WHY: Generalizes semantic entropy with kernel methods and von Neumann entropy
    - READ: Sections 1-3 (skip heavy math in Section 4 on first read)
    - TIME: 2-3 hours
    - TAKEAWAY: Theoretical unification of entropy methods

### Embeddings and Semantic Similarity
14. **Reimers, N., & Gurevych, I. (2019). Sentence-BERT**
    - WHY: Foundation for our semantic similarity calculations in MI estimation
    - READ: Sections 1-3
    - TIME: 1-2 hours
    - TAKEAWAY: How to efficiently compute semantic similarity

15. **OpenAI (2024). New Embedding Models (Blog Post)**
    - WHY: Introduces text-embedding-3 used in this research
    - READ: Full blog post
    - TIME: 30 minutes
    - TAKEAWAY: Performance improvements and Matryoshka representations

### Evaluation Metrics
16. **Liu, Y., et al. (2023). G-EVAL: Using GPT-4 for Evaluation**
    - WHY: Modern LLM-based evaluation, relevant for quality assessment
    - READ: Sections 1-4
    - TIME: 1-2 hours
    - TAKEAWAY: LLMs can judge quality better than traditional metrics

17. **Zhao, W., et al. (2023). DiscoScore (EACL)**
    - WHY: BERT-based coherence evaluation - relevant for our coherence metric
    - READ: Sections 1-3
    - TIME: 1-2 hours
    - TAKEAWAY: Discourse coherence via transformers

---

## TIER 3: Recommended for Depth

### Information Theory Applications
18. **Polyanskiy, Y., & Wu, Y. (2019). Fano's Inequality Guide**
    - WHY: Modern treatment of information bounds
    - READ: Sections 1-2, skim applications
    - TIME: 2-3 hours

19. **Kullback, S., & Leibler, R. A. (1951). On Information and Sufficiency**
    - WHY: Original KL divergence paper
    - READ: Sections 1-3
    - TIME: 1-2 hours (historical, dense)

### Advanced Prompting
20. **Wang, X., et al. (2022). Self-Consistency**
    - WHY: Shows ensemble approach improves CoT
    - READ: Full paper
    - TIME: 1 hour

21. **Yao, S., et al. (2023). Tree of Thoughts**
    - WHY: Advanced reasoning framework
    - READ: Sections 1-4
    - TIME: 1-2 hours

22. **Kojima, T., et al. (2022). Zero-Shot Reasoners**
    - WHY: "Let's think step by step" - shows phrasing matters
    - READ: Full paper
    - TIME: 1 hour

### Model Training and Alignment
23. **Ouyang, L., et al. (2022). InstructGPT**
    - WHY: RLHF for instruction following
    - READ: Sections 1-3
    - TIME: 2 hours

24. **Bai, Y., et al. (2022). Constitutional AI**
    - WHY: Alignment method used in Claude
    - READ: Sections 1-3
    - TIME: 2 hours

### Survey Papers
25. **Sahoo, P., et al. (2024). Systematic Survey of Prompt Engineering**
    - WHY: Comprehensive overview of techniques
    - READ: Skim entire, deep read on relevant sections
    - TIME: 3-4 hours

26. **Gao, Y., et al. (2024). RAG Survey**
    - WHY: Retrieval-augmented generation context
    - READ: Sections 1-3
    - TIME: 2-3 hours

---

## TIER 4: Specialized Topics

### Temperature Optimization
27. **Zhang, L., et al. (2025). Optimizing Temperature**
    - WHY: Recent work on temperature selection
    - TIME: 1-2 hours

28. **Fan, A., et al. (2018). Hierarchical Story Generation**
    - WHY: Temperature in creative tasks
    - TIME: 1-2 hours

### Uncertainty Quantification
29. **Guan, W., et al. (2024). UQ for Hallucination Detection**
    - WHY: Survey of UQ methods
    - TIME: 2-3 hours

30. **Chen, W., et al. (2024). LM-Polygraph Benchmark**
    - WHY: Comprehensive UQ benchmark
    - TIME: 2 hours

### Safety and Robustness
31. **Zou, A., et al. (2023). Universal Adversarial Attacks**
    - WHY: Jailbreaking attacks
    - TIME: 1-2 hours

32. **Robey, A., et al. (2023). SmoothLLM**
    - WHY: Defense mechanisms
    - TIME: 1-2 hours

### Classic Evaluation
33. **Papineni, K., et al. (2002). BLEU**
    - WHY: Classic n-gram metric
    - TIME: 1 hour

34. **Lin, C.-Y. (2004). ROUGE**
    - WHY: Recall-oriented metrics
    - TIME: 1 hour

---

## Reading Plans by Goal

### Goal: Understand Core Research (Minimum)
**Time: ~20-25 hours**
1. Shannon (1948) - 4 hours
2. Cover & Thomas Ch. 2 - 4 hours
3. Kuhn et al. (2023) - 3 hours
4. Kuhn et al. (2024) - 2 hours
5. Robinson et al. (2024) - 2 hours
6. Wei et al. (2022) - 2 hours
7. White et al. (2023) - 3 hours

### Goal: Extend Research (Comprehensive)
**Time: ~40-50 hours**
- All Tier 1 (8 papers, ~20 hours)
- All Tier 2 (9 papers, ~15 hours)
- Selected Tier 3 (5-7 papers, ~10-15 hours)

### Goal: Become Domain Expert (Complete)
**Time: ~80-100 hours**
- All Tier 1-3 (34 papers, ~60 hours)
- Selected Tier 4 (8 papers, ~15 hours)
- Deep dive into Cover & Thomas full book (~25 hours)

### Goal: Replicate Study
**Time: ~15-20 hours**
1. Kuhn et al. (2023, 2024) - semantic entropy implementation
2. Robinson et al. (2024) - temperature study design
3. White et al. (2023) - prompt patterns
4. OpenAI (2024) - embedding API
5. Liu et al. (2023) or Zhao et al. (2023) - quality evaluation

### Goal: Write Literature Review
**Time: ~30-35 hours**
- Tier 1: Full depth (8 papers, ~20 hours)
- Tier 2: Good understanding (9 papers, ~12 hours)
- Tier 3+4: Skim for citations (20 papers, ~3-5 hours)

---

## Reading Order Recommendations

### Order 1: Theory First
Best for: Researchers with math/CS background
1. Shannon (1948)
2. Cover & Thomas (2006)
3. Kuhn et al. (2023)
4. Kuhn et al. (2024)
5. Robinson et al. (2024)
→ Continue with Tier 2

### Order 2: Practice First
Best for: Practitioners, engineers
1. White et al. (2023)
2. Wei et al. (2022)
3. Kuhn et al. (2024 Nature)
4. Robinson et al. (2024)
5. Shannon (1948)
6. Kuhn et al. (2023)
→ Continue with Tier 2

### Order 3: Chronological
Best for: Understanding field evolution
1. Shannon (1948)
2. Kullback & Leibler (1951)
3. Vaswani et al. (2017)
4. Holtzman et al. (2019)
5. Brown et al. (2020)
6. Wei et al. (2022)
7. Kuhn et al. (2023)
8. Achiam et al. (2023)
9. Robinson et al. (2024)
10. Kuhn et al. (2024)
→ Continue chronologically

### Order 4: Impact-Based
Best for: Quick understanding of key ideas
1. Shannon (1948) - foundational
2. Vaswani et al. (2017) - 100k+ citations
3. Brown et al. (2020) - 20k+ citations
4. Wei et al. (2022) - revolutionized prompting
5. Kuhn et al. (2024) - Nature paper
→ Continue with high-impact papers

---

## Quick Reference Cards

### For Writing Introduction:
- Shannon (1948): foundational theory
- Brown (2020): few-shot learning context
- Wei (2022): CoT prompting importance
- White (2023): practical prompt engineering
- Kuhn (2024): entropy for quality assessment

### For Writing Methods:
- Kuhn (2023): semantic entropy calculation
- Robinson (2024): temperature study design
- Reimers (2019): embedding similarity
- Holtzman (2019): sampling strategies
- OpenAI (2024): embedding models

### For Writing Results:
- Kuhn (2024): effect sizes for entropy
- Robinson (2024): temperature effect analysis
- Wei (2022): specification effect sizes
- Liu (2023): quality evaluation methods

### For Writing Discussion:
- Cover & Thomas (2006): theoretical implications
- Kuhn (2024): practical applications
- Sahoo (2024): broader context
- Gao (2024): future directions (RAG)

---

## Key Concepts by Paper

| Paper | Key Concept | Formula/Metric |
|-------|-------------|----------------|
| Shannon (1948) | Entropy | H(X) = -Σ p(x) log p(x) |
| Shannon (1948) | Mutual Info | I(X;Y) = H(X) - H(X\|Y) |
| Kuhn (2023) | Semantic Entropy | Cluster → P(meaning) → H |
| Robinson (2024) | Temp Effect | Performance vs T ∈ [0,2] |
| Wei (2022) | CoT | Few-shot + reasoning steps |
| Holtzman (2019) | Nucleus | Top-p cumulative probability |
| Brown (2020) | Few-shot | k examples → performance |

---

## Paper Difficulty Ratings

### Beginner-Friendly:
- White (2023) - prompt patterns ⭐
- Wei (2022) - chain-of-thought ⭐
- Robinson (2024) - temperature study ⭐⭐
- Kuhn (2024 Nature) - semantic entropy ⭐⭐

### Intermediate:
- Brown (2020) - GPT-3 ⭐⭐
- Kuhn (2023) - semantic uncertainty ⭐⭐⭐
- Holtzman (2019) - sampling ⭐⭐⭐
- Reimers (2019) - SBERT ⭐⭐⭐

### Advanced:
- Shannon (1948) - information theory ⭐⭐⭐⭐
- Cover & Thomas (2006) - textbook ⭐⭐⭐⭐
- Vaswani (2017) - transformers ⭐⭐⭐⭐
- Lin (2024) - kernel entropy ⭐⭐⭐⭐⭐

---

## Implementation Priority

If implementing from scratch:
1. **Kuhn (2023)** - semantic entropy core algorithm
2. **Reimers (2019)** - embedding similarity
3. **OpenAI (2024)** - embedding API
4. **Robinson (2024)** - temperature experimental design
5. **White (2023)** - prompt template design
6. **Liu (2023)** - quality evaluation

---

## Citation Density Map

Papers to cite multiple times:
- Shannon (1948): 5-10 times (fundamental concepts)
- Cover & Thomas (2006): 5-8 times (theoretical predictions)
- Kuhn et al. (2023, 2024): 8-12 times (core methodology)
- Robinson et al. (2024): 3-5 times (temperature validation)
- Wei et al. (2022): 2-4 times (prompt engineering context)

Papers to cite 1-2 times:
- Most Tier 2-4 papers (supporting evidence, related work)

---

## Online Resources

### Accompanying Materials:
- **Cover & Thomas**: Solutions manual available
- **Kuhn et al. (2023)**: Code on GitHub
- **Sahoo et al. (2024)**: Comprehensive bibliography
- **OpenAI (2024)**: API documentation and examples

### Tutorials:
- **3Blue1Brown**: Information theory visualization (YouTube)
- **Hugging Face**: Transformer tutorials
- **Fast.ai**: Practical deep learning

---

**Last Updated:** 2025-01-19

**Total Reading Time:**
- Tier 1 (Essential): ~20-25 hours
- Tier 2 (Recommended): +15 hours
- Tier 3 (Depth): +20 hours
- Tier 4 (Specialized): +15 hours
- **Complete mastery: ~70-80 hours**

**Quick Start (8 hours):**
Shannon (1948) → Kuhn (2024 Nature) → Robinson (2024) → White (2023)
