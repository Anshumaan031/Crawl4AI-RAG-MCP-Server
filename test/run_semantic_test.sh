#!/bin/bash

# Check if Python is installed
if ! command -v python &>/dev/null; then
    echo "Python is not installed. Please install Python 3.x."
    exit 1
fi

# Check if virtualenv is installed
if ! command -v pip &>/dev/null; then
    echo "pip is not installed. Please install pip."
    exit 1
fi

# Set environment variables
if [ ! -f .env ]; then
    echo "Creating .env file for API keys..."
    echo "GOOGLE_API_KEY=YOUR_GEMINI_API_KEY" > .env
    echo ".env file created. Please edit it to add your Gemini API key."
    exit 1
fi

# Check if required packages are installed
echo "Installing required packages if needed..."
pip install llama-index google-generativeai python-dotenv langchain &>/dev/null

# Run the test
echo "Running semantic chunking test..."
python test_semantic_chunking.py

echo "Done!" 