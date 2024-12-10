# Variables
PYTHON = python
PIP = pip
SRC_DIR = src
DATASET_SCRIPT = $(SRC_DIR)/data/dataset.py
ENV_NAME = tts_torch

# Phony targets to avoid conflict with files of the same name
.PHONY: all setup env install clean test load_raw preprocess

# Default target
all: setup

# Set up the environment and install dependencies
setup: env install

# Create a Python virtual environment
env:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(ENV_NAME)
	@echo "Virtual environment created in $(ENV_NAME)"

# Install required packages
install:
	@echo "Installing dependencies..."
	$(ENV_NAME)/bin/$(PIP) install -r requirements.txt
	@echo "Dependencies installed!"

# Load raw dataset
load_raw:
	@echo "Loading raw dataset..."
	$(ENV_NAME)/bin/$(PYTHON) $(DATASET_SCRIPT) load_raw
	@echo "Raw dataset loaded!"

# Preprocess dataset
preprocess:
	@echo "Preprocessing dataset..."
	$(ENV_NAME)/bin/$(PYTHON) $(DATASET_SCRIPT) preprocess
	@echo "Dataset preprocessed!"

# Run tests
test:
	@echo "Running tests..."
	$(ENV_NAME)/bin/$(PYTHON) -m pytest tests
	@echo "Tests completed!"

# Clean up generated files and artifacts
clean:
	@echo "Cleaning up..."
	rm -rf $(ENV_NAME) data/raw/* data/processed/*
	@echo "Cleanup complete!"

dataset:
	@echo "Creating dataset..."
	$(ENV_NAME)/bin/$(PYTHON) $(DATASET_SCRIPT) create
	@echo "Dataset created!"
