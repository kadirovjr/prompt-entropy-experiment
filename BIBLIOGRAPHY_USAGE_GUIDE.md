# Bibliography Usage Guide

Quick guide for using the comprehensive bibliography in your research.

---

## Files Overview

### 1. **`paper/references_comprehensive.bib`** - BibTeX Database

**Use for:** LaTeX paper compilation

```latex
\bibliographystyle{plain}
\bibliography{references_comprehensive}
```

### 2. **`BIBLIOGRAPHY.md`** - Annotated Bibliography
**Use for:** Understanding paper relevance before reading
- Full descriptions of each paper
- Organized by 12 categories
- Includes key concepts and relevance notes

### 3. **`REFERENCES_BY_TOPIC.md`** - Topic-Organized References
**Use for:** Finding papers for specific research questions
- Organized by hypothesis (H1-H4)
- Grouped by methodology needs
- Quick lookup by year

### 4. **`RECOMMENDED_READING.md`** - Prioritized Reading List
**Use for:** Planning your reading schedule
- 4 tiers of priority
- Time estimates for each paper
- Multiple reading order suggestions

### 5. **`LITERATURE_REVIEW_SUMMARY.md`** - Executive Summary
**Use for:** Quick overview and research gaps
- Key statistics
- Research timeline
- Citation strategy

---

## Quick Workflows

### Workflow 1: Writing Literature Review
```
1. Read LITERATURE_REVIEW_SUMMARY.md (30 min)
2. Skim BIBLIOGRAPHY.md categories (2 hours)
3. Read Tier 1 papers from RECOMMENDED_READING.md (20 hours)
4. Draft related work section
5. Use REFERENCES_BY_TOPIC.md to find supporting citations
6. Add citations to paper/references_comprehensive.bib
```

### Workflow 2: Writing Specific Paper Section
```
Example: Writing "Methods - Semantic Entropy"

1. Open REFERENCES_BY_TOPIC.md
2. Find "For Semantic Entropy Metric" section
3. Read core papers:
   - Kuhn et al. (2023) - original method
   - Kuhn et al. (2024) - validation
4. Cite in your LaTeX:
   \cite{kuhn2023semantic,kuhn2024detecting}
5. BibTeX handles formatting automatically
```

### Workflow 3: Finding Papers for Hypothesis
```
Example: Supporting H4 (MI-Quality Correlation)

1. Open REFERENCES_BY_TOPIC.md
2. Navigate to "For H4: Mutual Information Correlates with Quality"
3. Read must-read papers:
   - Shannon (1948) - MI definition
   - Kuhn et al. (2024) - entropy-quality link
   - Liu et al. (2023) - quality assessment
4. Add to paper with: \cite{shannon1948mathematical,kuhn2024detecting,liu2023geval}
```

### Workflow 4: Quick Literature Search
```
Need a paper on: "Temperature effects in LLMs"

1. Open BIBLIOGRAPHY.md
2. Ctrl+F: "temperature"
3. Find relevant papers:
   - Robinson et al. (2024) - systematic study
   - Holtzman et al. (2019) - sampling
   - Zhang et al. (2025) - optimization
4. Read abstracts/summaries
5. Decide which to read fully
```

---

## Using the BibTeX File

### In Your LaTeX Paper:

**Step 1:** Ensure file is in correct location
```bash
# Should be at:
paper/references_comprehensive.bib
```

**Step 2:** Update your .tex file
```latex
% At end of document, before \end{document}

\bibliographystyle{plain}  % or ieeetr, acm, etc.
\bibliography{references_comprehensive}

\end{document}
```

**Step 3:** Cite papers in text
```latex
Shannon entropy \cite{shannon1948mathematical} measures...

Recent work on semantic entropy \cite{kuhn2023semantic,kuhn2024detecting}
has shown...

Following Cover and Thomas \cite{cover2006elements}, we define...
```

**Step 4:** Compile (run multiple times)
```bash
cd paper/
pdflatex prompt_entropy_paper.tex
bibtex prompt_entropy_paper
pdflatex prompt_entropy_paper.tex
pdflatex prompt_entropy_paper.tex
```

### Common Citation Patterns:

**Single author:**
```latex
\cite{shannon1948mathematical}
```
Output: [1] or (Shannon, 1948) depending on style

**Multiple works:**
```latex
\cite{shannon1948mathematical,cover2006elements}
```
Output: [1,2] or (Shannon, 1948; Cover & Thomas, 2006)

**In-text citation:**
```latex
As shown by \citet{kuhn2024detecting}, semantic entropy...
```
(Requires natbib package)

**Parenthetical citation:**
```latex
Semantic entropy predicts quality \citep{kuhn2024detecting}.
```
(Requires natbib package)

---

## Finding Papers by Need

### Need: Theoretical Foundation
**Source:** BIBLIOGRAPHY.md → Category 1
**Top 3:**
1. Shannon (1948) - H(X) and I(X;Y)
2. Cover & Thomas (2006) - comprehensive reference
3. Polyanskiy & Wu (2019) - modern treatment

**Quick cite:**
```latex
\cite{shannon1948mathematical,cover2006elements}
```

### Need: Semantic Entropy Implementation
**Source:** REFERENCES_BY_TOPIC.md → "For Semantic Entropy Metric"
**Top 3:**
1. Kuhn et al. (2023) - method description
2. Kuhn et al. (2024) - validation
3. Farquhar et al. (2024) - improvements

**Quick cite:**
```latex
\cite{kuhn2023semantic,kuhn2024detecting,farquhar2024beyond}
```

### Need: Temperature Study Design
**Source:** REFERENCES_BY_TOPIC.md → "For H2: Entropy Increases with Temperature"
**Top 3:**
1. Robinson et al. (2024) - systematic study
2. Holtzman et al. (2019) - sampling
3. Zhang et al. (2025) - optimization

**Quick cite:**
```latex
\cite{robinson2024effect,holtzman2019curious,zhang2025optimizing}
```

### Need: Prompt Engineering Context
**Source:** BIBLIOGRAPHY.md → Category 3
**Top 3:**
1. White et al. (2023) - pattern catalog
2. Wei et al. (2022) - chain-of-thought
3. Sahoo et al. (2024) - comprehensive survey

**Quick cite:**
```latex
\cite{white2023prompt,wei2022chain,sahoo2024systematic}
```

---

## Reading Schedule Templates

### Template 1: Weekend Deep Dive (12 hours)
**Goal:** Core understanding for paper writing

**Saturday (6 hours):**
- 09:00-11:00: Kuhn et al. (2024 Nature) - 2h
- 11:00-13:00: Robinson et al. (2024) - 2h
- 14:00-16:00: White et al. (2023) - 2h

**Sunday (6 hours):**
- 09:00-13:00: Shannon (1948) - 4h
- 14:00-16:00: Wei et al. (2022) - 2h

**Outcome:** Ready to write Methods and Results

### Template 2: Week-Long Intensive (25 hours)
**Goal:** Comprehensive understanding

**Monday (4h):** Shannon (1948)
**Tuesday (4h):** Cover & Thomas Ch. 2
**Wednesday (5h):** Kuhn (2023, 2024)
**Thursday (4h):** Robinson (2024) + Holtzman (2019)
**Friday (6h):** Wei (2022) + White (2023) + Sahoo (2024)
**Weekend (2h):** Review and notes

**Outcome:** Ready to write full paper with confidence

### Template 3: Gradual Build (4 weeks, 5h/week)
**Goal:** Deep understanding while working on other tasks

**Week 1 (5h):** Information theory
- Shannon (1948) - 3h
- Cover & Thomas Ch. 2 - 2h

**Week 2 (5h):** Semantic entropy
- Kuhn (2023) - 3h
- Kuhn (2024) - 2h

**Week 3 (5h):** Temperature and sampling
- Robinson (2024) - 2h
- Holtzman (2019) - 1.5h
- Zhang (2025) - 1.5h

**Week 4 (5h):** Prompt engineering
- Wei (2022) - 2h
- White (2023) - 2h
- Sahoo (2024) skim - 1h

**Total: 20 hours over 4 weeks**

---

## Citation Strategy by Paper Section

### Abstract (0 citations)
No citations in abstract

### Introduction (8-10 citations)
```latex
Prompt engineering has emerged as critical \cite{white2023prompt},
yet lacks mathematical foundations. We address this using
information theory \cite{shannon1948mathematical,cover2006elements}.

Recent work shows entropy predicts quality \cite{kuhn2024detecting}...
```

**Papers:**
- shannon1948mathematical
- cover2006elements
- white2023prompt
- wei2022chain
- brown2020language
- kuhn2024detecting
- sahoo2024systematic

### Background (15-20 citations)

**Theory section:**
```latex
Shannon entropy \cite{shannon1948mathematical} measures uncertainty.
The data processing inequality \cite{cover2006elements} states that
I(X;Y) >= I(X;f(Y))...
```

**LLMs section:**
```latex
Modern LLMs \cite{brown2020language,achiam2023gpt4} use the
Transformer architecture \cite{vaswani2017attention}...
```

**Entropy section:**
```latex
Semantic entropy \cite{kuhn2023semantic,kuhn2024detecting}
extends token-based measures...
```

### Methods (10-15 citations)

**Semantic entropy:**
```latex
Following Kuhn et al. \cite{kuhn2023semantic}, we cluster
responses using bidirectional entailment...
```

**Temperature:**
```latex
We test across three temperatures \cite{robinson2024effect}:
T=0.7 (production), T=1.0 (baseline), T=1.2 (exploration)...
```

**Embeddings:**
```latex
We use text-embedding-3-small \cite{openai2024embeddings}
for semantic similarity \cite{reimers2019sentence}...
```

### Results (15-20 citations)
Compare with related work, support findings

### Discussion (10-15 citations)
Theoretical implications, practical guidelines, limitations, future work

### Conclusion (2-3 citations)
Summary citations only

**Total: 60-80 citations recommended**

---

## Advanced BibTeX Usage

### Using Natbib for Better Citations

**Add to preamble:**
```latex
\usepackage{natbib}
\bibliographystyle{plainnat}  % or abbrvnat
```

**Citation commands:**
```latex
\citet{shannon1948mathematical}     → Shannon (1948)
\citep{shannon1948mathematical}     → (Shannon, 1948)
\citet*{brown2020language}          → Brown et al. (2020)
\citep{kuhn2023semantic,kuhn2024detecting} → (Kuhn et al., 2023, 2024)
```

### Custom Bibliography Styles

**IEEE style:**
```latex
\bibliographystyle{IEEEtran}
```

**ACM style:**
```latex
\bibliographystyle{ACM-Reference-Format}
```

**Nature style:**
```latex
\bibliographystyle{naturemag}
```

### Selective Bibliography

If you want only certain categories:

**Create subset file:**
```bash
# Extract only Tier 1 papers
grep -A 20 "Category 1\|Category 2" references_comprehensive.bib > references_tier1.bib
```

---

## Verification Checklist

Before submitting paper:

### Bibliography Complete:
- [ ] All cited papers in .bib file
- [ ] BibTeX compiled without errors
- [ ] All references appear in PDF
- [ ] No "?" citations in output

### Citation Quality:
- [ ] Key claims cited
- [ ] Methods properly attributed
- [ ] Related work comprehensively covered
- [ ] Recent work (2023-2025) included

### Formatting:
- [ ] Author names correct
- [ ] Titles capitalized properly
- [ ] Years correct
- [ ] Venue information complete

### Coverage:
- [ ] Information theory foundations cited
- [ ] LLM architecture papers cited
- [ ] Semantic entropy work cited
- [ ] Temperature studies cited
- [ ] Prompt engineering cited

---

## Common Issues and Solutions

### Issue: Citation not appearing
**Solution:**
```bash
# Check compilation order:
pdflatex file.tex      # First pass
bibtex file            # Process bibliography
pdflatex file.tex      # Second pass
pdflatex file.tex      # Third pass
```

### Issue: BibTeX entry malformed
**Solution:**
- Check for missing commas
- Ensure all braces balanced: `{}`
- Verify required fields present
- Use our pre-validated .bib file

### Issue: Too many/few citations
**Solution:**
- Aim for 60-80 total
- 8-10 in Introduction
- 15-20 in Background
- 10-15 in Methods
- 15-20 in Results
- 10-15 in Discussion

### Issue: Missing recent papers
**Solution:**
- Check BIBLIOGRAPHY.md "Recent Advances (2024)" section
- Priority: Kuhn (2024), Robinson (2024), Farquhar (2024)

---

## Quick Reference Card

**Find papers by hypothesis:**
→ REFERENCES_BY_TOPIC.md

**Find papers by category:**
→ BIBLIOGRAPHY.md

**Find reading priority:**
→ RECOMMENDED_READING.md

**Find BibTeX entries:**
→ paper/references_comprehensive.bib

**Find overview:**
→ LITERATURE_REVIEW_SUMMARY.md

**Find this guide:**
→ BIBLIOGRAPHY_USAGE_GUIDE.md

---

## Updating the Bibliography

### Adding New Papers:

1. Find paper details
2. Add to `references_comprehensive.bib`:
```bibtex
@article{author2025title,
  title={Paper Title},
  author={Author, First and Author, Second},
  journal={Venue},
  year={2025},
  note={Brief relevance note}
}
```

3. Update relevant markdown files:
   - Add to BIBLIOGRAPHY.md (appropriate category)
   - Add to REFERENCES_BY_TOPIC.md (relevant sections)
   - Consider for RECOMMENDED_READING.md if highly relevant

### Removing Papers:

Only if not cited in final paper:
1. Comment out in .bib file: `% @article{...}`
2. Note removal in markdown files

---

## Tools and Software

### Recommended Bibliography Managers:
- **Zotero** (Free, cross-platform) - Best for beginners
- **Mendeley** (Free) - Good PDF annotation
- **BibDesk** (Free, Mac) - Native .bib support
- **JabRef** (Free, cross-platform) - Java-based

### Importing Our Bibliography:
1. Open bibliography manager
2. File → Import → BibTeX
3. Select `references_comprehensive.bib`
4. Organize into folders by category

### PDF Management:
- Store PDFs in `/papers/` directory
- Name: `author_year_title.pdf`
- Link to BibTeX entries in manager

---

## Time Estimates

### Using Bibliography for Paper Writing:

**Quick cite lookup:** 2-5 minutes per citation
**Finding related work:** 30-60 minutes
**Writing literature review:** 8-12 hours (with reading)
**Integrating citations:** 4-6 hours
**Formatting bibliography:** 1-2 hours

**Total for full paper:** 15-25 hours bibliography work

### Reading Papers:

**Skim:** 15-30 minutes per paper
**Read:** 1-3 hours per paper
**Deep study:** 4-8 hours per paper

---

## Contact and Support

**Questions about bibliography?**
- Check this guide first
- Review relevant markdown file
- Consult paper abstracts in BIBLIOGRAPHY.md

**Need specific papers?**
- Most available on arXiv (free)
- ACL Anthology (free)
- Library access for journals

**Want to extend research?**
- See "Future Work" in LITERATURE_REVIEW_SUMMARY.md
- Check Tier 3-4 in RECOMMENDED_READING.md

---

**Created:** 2025-01-19
**Version:** 1.0
**Maintainer:** Ibrahim Cesar
**Repository:** https://github.com/ibrahimcesar/prompt-entropy-experiment
