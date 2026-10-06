.PHONY: check test lint coverage clean demo-repro bench-attest campaigns

check: test check-imports

campaigns: campaign-mutation campaign-crash campaign-divergence

campaign-mutation:
	uv run python -m campaigns.mutation

campaign-crash:
	uv run python -m campaigns.crash

campaign-divergence:
	uv run python -m campaigns.divergence

demo:
	uv run python -m demo.tiny_dqn

demo-repro:
	uv run python -m demo.reproducible_grpo

bench-attest:
	uv run python -m benchmarks.attestation_overhead

tla:
	bash spec/check.sh

test:
	uv run pytest tests/ -v --tb=short

coverage:
	uv run pytest tests/ --cov=src/reservoir --cov-report=term-missing --cov-report=html

lint:
	uv run python -m py_compile src/reservoir/*.py checker/*.py

check-imports:
	uv run --no-sync python -c "import ast, os, sys; \
	d = 'src/reservoir_checker'; \
	names = [(f, (getattr(n, 'module', None) or n.names[0].name)) for f in os.listdir(d) if f.endswith('.py') \
	for n in ast.walk(ast.parse(open(os.path.join(d, f)).read())) if isinstance(n, (ast.Import, ast.ImportFrom))]; \
	bad = [(f, m) for f, m in names if m == 'reservoir' or m.startswith('reservoir.')]; \
	sys.exit('checker imports reservoir: %s' % bad) if bad else print('checker import isolation ok')"

clean:
	rm -rf .pytest_cache __pycache__ .coverage htmlcov
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
