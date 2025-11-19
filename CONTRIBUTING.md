# Contributing to Prompt Entropy Experiment

Thank you for your interest in contributing to this research project!

## Getting Started

1. **Fork the repository** and clone your fork
2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
├── paper/              # Research paper (LaTeX)
├── data/              # Experimental data
├── notebooks/         # Analysis notebooks
├── src/               # Core analysis code
│   ├── metrics/       # Entropy, MI, quality calculations
│   ├── analysis/      # Statistical analysis
│   └── utils/         # Helper functions
└── tests/             # Unit tests
```

## Development Workflow

### Adding New Metrics

1. Add implementation to appropriate module in `src/metrics/`
2. Add tests to `tests/`
3. Document in module docstring
4. Update `__init__.py` exports

### Running Tests

```bash
pytest tests/
pytest --cov=src tests/  # With coverage
```

### Code Style

- Follow PEP 8
- Use type hints
- Write docstrings for all public functions
- Run `mypy src/` for type checking

## Research Contributions

### Experimental Data

- Store raw data in `data/raw/`
- Store processed data in `data/processed/`
- Document data collection in notebooks
- Update paper with new findings

### Analysis Notebooks

- Use clear markdown documentation
- Include visualizations
- Show statistical results
- Reference paper sections

### Paper Updates

- Edit `paper/prompt_entropy_paper.tex`
- Follow arXiv formatting guidelines
- Compile with: `pdflatex prompt_entropy_paper.tex`
- Cite all sources properly

## Submitting Changes

1. **Create a branch**: `git checkout -b feature/your-feature`
2. **Make your changes** with clear commit messages
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Submit a pull request** with description of changes

## Questions?

Open an issue or contact: ibrahim@ibrahimcesar.com

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
