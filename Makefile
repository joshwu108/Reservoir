.PHONY: check test lint coverage clean

check: test

test:
	uv run pytest tests/ -v --tb=short

coverage:
	uv run pytest tests/ --cov=src/reservoir --cov-report=term-missing --cov-report=html

lint:
	uv run python -m py_compile src/reservoir/*.py checker/*.py

check-imports:
	uv run python -c "import ast, sys; \
	import os; \
	[sys.exit('checker imports src/reservoir!') for f in os.listdir('checker') if f.endswith('.py') \
	for node in ast.walk(ast.parse(open('checker/'+f).read())) \
	if isinstance(node, (ast.Import, ast.ImportFrom)) \
	and getattr(node, 'module', '') and 'reservoir' in getattr(node, 'module', '')]"

clean:
	rm -rf .pytest_cache __pycache__ .coverage htmlcov
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
