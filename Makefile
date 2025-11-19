.PHONY: help install test lint paper clean

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run linting"
	@echo "  make paper      - Compile LaTeX paper"
	@echo "  make clean      - Clean generated files"
	@echo "  make notebook   - Start Jupyter notebook server"

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest tests/ -v

test-cov:
	pytest --cov=src tests/ --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

lint:
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/

paper:
	cd paper && pdflatex prompt_entropy_paper.tex
	cd paper && pdflatex prompt_entropy_paper.tex  # Run twice for references
	@echo "Paper compiled to paper/prompt_entropy_paper.pdf"

notebook:
	jupyter notebook notebooks/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	cd paper && rm -f *.aux *.log *.out *.toc *.synctex.gz *.fdb_latexmk *.fls
	@echo "Cleaned generated files"

setup-dev:
	pip install -e ".[dev]"
	@echo "Development environment ready!"
