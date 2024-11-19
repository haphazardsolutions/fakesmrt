#!/bin/bash
# Setup script for fakesmrt development environment

# Exit on any error
set -e

echo "Setting up fakesmrt development environment..."

# Check if we're in the right directory (should contain setup.sh)
if [ ! -f "setup.sh" ]; then
    echo "Error: Please run this script from the repository root directory"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.9"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "Error: Python $REQUIRED_VERSION or higher is required (found $PYTHON_VERSION)"
    exit 1
fi

# Create and activate virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip and install basic tools
echo "Upgrading pip and installing basic tools..."
pip install --upgrade pip
pip install wheel setuptools

# Install requirements
echo "Installing project dependencies..."
pip install -r requirements.txt

# Verify required directories exist
echo "Verifying project structure..."
for dir in src tests data models configs; do
    if [ ! -d "$dir" ]; then
        echo "Creating $dir directory..."
        mkdir -p "$dir"
    fi
done

# Verify required files exist
for file in README.md .gitignore; do
    if [ ! -f "$file" ]; then
        echo "Warning: $file not found. Please ensure it exists."
    fi
done

# Set up pre-commit hooks if git is available
if command -v git >/dev/null 2>&1; then
    if [ -d ".git" ]; then
        echo "Setting up git hooks..."
        if [ ! -f ".git/hooks/pre-commit" ]; then
            cat > .git/hooks/pre-commit << 'EOL'
#!/bin/bash
# Run tests before commit
python -m pytest tests/
EOL
            chmod +x .git/hooks/pre-commit
        fi
    fi
fi

# Run initial tests if pytest is available
if python -c "import pytest" >/dev/null 2>&1; then
    echo "Running initial tests..."
    python -m pytest tests/
fi

echo "Setup complete! Activate the virtual environment with:"
echo "    source venv/bin/activate"
