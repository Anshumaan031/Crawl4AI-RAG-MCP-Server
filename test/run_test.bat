@echo off
echo Running Semantic Chunking Test

REM Check if Python is installed
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Python is not installed or not in PATH. Please install Python 3.x.
    exit /b 1
)

REM Check if .env file exists
if not exist .env (
    echo Creating sample .env file...
    echo GEMINI_API_KEY=your_gemini_api_key_here > .env
    echo OPENAI_API_KEY=your_openai_api_key_here >> .env
    echo .env file created. Please edit it to add your API keys before running the test.
    exit /b 1
)

REM Install required packages
echo Installing required packages...
pip install -q llama-index google-generativeai python-dotenv openai

REM Run the test
echo Running semantic chunking test...
python test_semantic_chunking.py

echo Done! 