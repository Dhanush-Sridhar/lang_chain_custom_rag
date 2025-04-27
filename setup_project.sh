#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 [-p|--path PATH]"
    echo "  -p, --path PATH    Specify the path where the project will be created"
    echo "                     If not specified, creates project in current directory"
    exit 1
}

# Parse command line arguments
PROJECT_PATH="."
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--path)
            PROJECT_PATH="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Function to create directory if it doesn't exist
create_directory() {
    if [ ! -d "$1" ]; then
        echo "Creating directory: $1"
        mkdir -p "$1"
    else
        echo "Directory already exists: $1"
    fi
}

# Create project directory if it doesn't exist
create_directory "$PROJECT_PATH"

# Change to project directory
cd "$PROJECT_PATH" || { echo "Failed to change to directory: $PROJECT_PATH"; exit 1; }

# Create project directory structure
echo "Creating project structure..."
mkdir -p src/data
mkdir -p src/models
mkdir -p src/utils
mkdir -p tests
mkdir -p docs

# Create basic Python files
touch src/__init__.py
touch src/data/__init__.py
touch src/models/__init__.py
touch src/utils/__init__.py
touch tests/__init__.py

# Create main application file
cat > src/main.py << 'EOL'
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os

def main():
    # Your RAG implementation will go here
    pass

if __name__ == "__main__":
    main()
EOL

# Create requirements.txt
cat > requirements.txt << 'EOL'
langchain==0.1.0
openai==1.3.0
faiss-cpu==1.7.4
python-dotenv==1.0.0
pytest==7.4.3
EOL

# Create README.md
cat > README.md << 'EOL'
# LangChain RAG Application

A simple RAG (Retrieval-Augmented Generation) application built using LangChain. This project serves as a learning exercise to understand and implement RAG concepts using the LangChain framework.

## Project Description

This application demonstrates how to:
- Implement a basic RAG pipeline using LangChain
- Process and store documents for retrieval
- Generate responses using retrieved context
- Combine language models with document retrieval

The project is designed to help developers understand the fundamentals of RAG systems and how to implement them using LangChain.

## Setup Instructions

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a .env file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
EOL

# Create .env file
cat > .env << 'EOL'
OPENAI_API_KEY=your_api_key_here
EOL

# Create .gitignore
cat > .gitignore << 'EOL'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Environment variables
.env

# Project specific
*.db
*.sqlite3
EOL

echo "Project setup complete in: $PROJECT_PATH"
echo "Please follow these steps:"
echo "1. Create and activate a virtual environment"
echo "2. Install dependencies using: pip install -r requirements.txt"
echo "3. Update the .env file with your OpenAI API key"
echo "4. Start coding in src/main.py" 