.PHONY: data test lint app check

data:
	python scripts/simulate_data.py

test:
	pytest

lint:
	ruff check src tests scripts

app:
	streamlit run app/Home.py

check: lint test
	python scripts/check_naming.py
