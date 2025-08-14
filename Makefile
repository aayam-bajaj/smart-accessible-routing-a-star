# Makefile for Smart Accessible Routing System

# Variables
PYTHON := python3
PIP := pip
FLASK := flask
TEST_FLAGS := -v

# Default target
.PHONY: help
help:
	@echo "Smart Accessible Routing System - Makefile"
	@echo ""
	@echo "Available commands:"
	@echo "  setup        Install dependencies"
	@echo "  dev-setup    Install development dependencies"
	@echo "  run          Run the development server"
	@echo "  test         Run tests"
	@echo "  lint         Run code linting"
	@echo "  format       Format code with black"
	@echo "  init-db      Initialize the database"
	@echo "  clean        Clean up temporary files"
	@echo "  help         Show this help message"

# Install dependencies
.PHONY: setup
setup:
	$(PIP) install -r requirements.txt

# Install development dependencies
.PHONY: dev-setup
dev-setup:
	$(PIP) install -r requirements-dev.txt

# Run the development server
.PHONY: run
run:
	$(FLASK) run

# Run tests
.PHONY: test
test:
	$(PYTHON) -m pytest $(TEST_FLAGS) tests/

# Run code linting
.PHONY: lint
lint:
	$(PYTHON) -m flake8 app/ tests/
	$(PYTHON) -m pylint app/ tests/

# Format code with black
.PHONY: format
format:
	$(PYTHON) -m black app/ tests/

# Initialize the database
.PHONY: init-db
init-db:
	$(PYTHON) init_db.py

# Clean up temporary files
.PHONY: clean
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.log" -delete
	rm -rf *.egg-info
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

# Install and run in one command
.PHONY: install-and-run
install-and-run: setup run

# Install dev dependencies and run tests
.PHONY: dev-test
dev-test: dev-setup test

# Show current git status
.PHONY: status
status:
	git status

# Commit changes
.PHONY: commit
commit:
	git add .
	git commit -m "Update project"

# Push to remote repository
.PHONY: push
push:
	git push origin main

# Pull from remote repository
.PHONY: pull
pull:
	git pull origin main