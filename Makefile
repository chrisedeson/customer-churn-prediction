.PHONY: help setup data train test run clean all

help:
	@echo "Customer Churn Prediction - Available Commands:"
	@echo ""
	@echo "  make setup    - Install dependencies"
	@echo "  make data     - Download and prepare data"
	@echo "  make train    - Train all models"
	@echo "  make test     - Run test predictions"
	@echo "  make run      - Start Streamlit app"
	@echo "  make all      - Run complete pipeline (setup, data, train)"
	@echo "  make clean    - Remove generated files"

setup:
	pip install -r requirements.txt

data:
	python -m src.data_prep

train:
	python -m src.train_models

test:
	python -m src.predict

clv:
	python -m src.clv_analysis

interpretability:
	python -m src.interpretability

run:
	streamlit run app.py

all: setup data train
	@echo "Pipeline complete. Run 'make run' to start the app."

clean:
	rm -rf data/processed/*.csv
	rm -rf models/*.pkl
	rm -rf __pycache__ src/__pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
