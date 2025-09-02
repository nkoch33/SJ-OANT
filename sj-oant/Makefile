# TMM Project Makefile
# Simple commands for common development tasks

.PHONY: clean clean-cache clean-logs clean-temp install test lint format setup help

# Default target
help:
	@echo "TMM Project Commands:"
	@echo "  setup       - Set up development environment"
	@echo "  clean       - Remove all cache files, logs, and temporary files"
	@echo "  clean-cache - Remove only Python cache files (__pycache__, .pyc)"
	@echo "  clean-logs  - Remove only log files"
	@echo "  clean-temp  - Remove only temporary files"
	@echo "  install     - Install project dependencies"
	@echo "  test        - Run tests (when available)"
	@echo "  lint        - Run code linting (when available)"
	@echo "  format      - Format code (when available)"
	@echo "  demo        - Run architecture demonstration"

# Full cleanup
clean:
	@echo "🧹 Running full cleanup..."
	@python scripts/cleanup.py

# Clean only Python cache files
clean-cache:
	@echo "🧹 Cleaning Python cache files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete
	@echo "✅ Python cache files cleaned"

# Clean only log files
clean-logs:
	@echo "🧹 Cleaning log files..."
	@find . -name "*.log" -delete 2>/dev/null || true
	@find . -type d -name "logs" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Log files cleaned"

# Clean only temporary files
clean-temp:
	@echo "🧹 Cleaning temporary files..."
	@find . -name "*~" -delete 2>/dev/null || true
	@find . -name "*.swp" -delete 2>/dev/null || true
	@find . -name "*.swo" -delete 2>/dev/null || true
	@find . -name ".DS_Store" -delete 2>/dev/null || true
	@echo "✅ Temporary files cleaned"

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	@pip install -r requirements.txt
	@echo "✅ Dependencies installed"

# Run architecture demo
demo:
	@echo "🚀 Running TMM architecture demonstration..."
	@python demo_architecture.py

# Run tests (placeholder for future)
test:
	@echo "🧪 Running tests..."
	@echo "   (Tests not yet implemented)"

# Run linting (placeholder for future)
lint:
	@echo "🔍 Running code linting..."
	@echo "   (Linting not yet configured)"

# Format code (placeholder for future)
format:
	@echo "✨ Formatting code..."
	@echo "   (Code formatting not yet configured)"

# Set up development environment
setup:
	@echo "🚀 Setting up development environment..."
	@python scripts/setup_dev.py
