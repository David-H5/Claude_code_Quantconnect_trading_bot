#!/bin/bash
# WSL Development Environment Setup Script
# Run this script after cloning or moving the project to WSL

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "  QuantConnect Trading Bot - WSL Setup"
echo "========================================"
echo ""

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"
echo ""

# 1. Check Python version
echo -e "${YELLOW}[1/7] Checking Python...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✓ $PYTHON_VERSION${NC}"
else
    echo -e "${RED}✗ Python 3 not found. Please install Python 3.10+${NC}"
    exit 1
fi

# 2. Create virtual environment
echo ""
echo -e "${YELLOW}[2/7] Setting up virtual environment...${NC}"
if [ -d "venv" ]; then
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
else
    python3 -m venv venv
    echo -e "${GREEN}✓ Created virtual environment${NC}"
fi

# Activate venv
source venv/bin/activate
echo -e "${GREEN}✓ Activated virtual environment${NC}"

# 3. Install dependencies
echo ""
echo -e "${YELLOW}[3/7] Installing dependencies...${NC}"
pip install --upgrade pip -q
pip install -r requirements.txt -q
pip install -e . -q 2>/dev/null || echo "  (No setup.py found, skipping editable install)"
echo -e "${GREEN}✓ Dependencies installed${NC}"

# 4. Install and configure pre-commit
echo ""
echo -e "${YELLOW}[4/7] Setting up pre-commit hooks...${NC}"
pip install pre-commit -q
if [ -f ".git/hooks/pre-commit" ]; then
    echo -e "${GREEN}✓ Pre-commit hooks already installed${NC}"
else
    pre-commit install
    echo -e "${GREEN}✓ Pre-commit hooks installed${NC}"
fi

# 5. Create .env file
echo ""
echo -e "${YELLOW}[5/7] Setting up environment file...${NC}"
if [ -f ".env" ]; then
    echo -e "${GREEN}✓ .env file already exists${NC}"
else
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${GREEN}✓ Created .env from .env.example${NC}"
        echo -e "${YELLOW}  ⚠ Remember to edit .env with your API credentials${NC}"
    else
        echo -e "${YELLOW}⚠ No .env.example found, skipping${NC}"
    fi
fi

# 6. Create backup directory
echo ""
echo -e "${YELLOW}[6/7] Creating backup directory...${NC}"
if [ -d ".backups" ]; then
    echo -e "${GREEN}✓ .backups directory already exists${NC}"
else
    mkdir -p .backups
    echo -e "${GREEN}✓ Created .backups directory${NC}"
fi

# 7. Configure git
echo ""
echo -e "${YELLOW}[7/7] Configuring git...${NC}"

# Set autocrlf
git config --global core.autocrlf input
echo -e "${GREEN}✓ Set core.autocrlf = input${NC}"

# Set fileMode
git config --global core.fileMode false
echo -e "${GREEN}✓ Set core.fileMode = false${NC}"

# Check git identity
GIT_NAME=$(git config --global user.name 2>/dev/null || echo "")
GIT_EMAIL=$(git config --global user.email 2>/dev/null || echo "")

if [ -z "$GIT_NAME" ] || [ -z "$GIT_EMAIL" ]; then
    echo ""
    echo -e "${YELLOW}Git identity not configured. Please enter your details:${NC}"

    if [ -z "$GIT_NAME" ]; then
        read -p "  Your name: " GIT_NAME
        git config --global user.name "$GIT_NAME"
    fi

    if [ -z "$GIT_EMAIL" ]; then
        read -p "  Your email: " GIT_EMAIL
        git config --global user.email "$GIT_EMAIL"
    fi
    echo -e "${GREEN}✓ Git identity configured${NC}"
else
    echo -e "${GREEN}✓ Git identity: $GIT_NAME <$GIT_EMAIL>${NC}"
fi

# Make scripts executable
chmod +x scripts/*.sh 2>/dev/null || true

# Summary
echo ""
echo "========================================"
echo -e "${GREEN}  Setup Complete!${NC}"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Edit .env with your API credentials:"
echo "     nano .env"
echo ""
echo "  2. Enable Docker Desktop WSL integration:"
echo "     Docker Desktop → Settings → Resources → WSL Integration"
echo ""
echo "  3. Activate the virtual environment in new terminals:"
echo "     source venv/bin/activate"
echo ""
echo "  4. Run tests to verify setup:"
echo "     pytest tests/ -v"
echo ""
echo "  5. (Optional) Open in VS Code:"
echo "     code ."
echo ""
