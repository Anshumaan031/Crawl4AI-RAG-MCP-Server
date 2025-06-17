@echo off
echo Running PDF Semantic Chunking Test

REM Check if Python is installed
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Python is not installed or not in PATH. Please install Python 3.x.
    exit /b 1
)

REM Check if the .env file exists
if not exist .env (
    echo Creating .env file for API keys...
    echo GOOGLE_API_KEY=YOUR_GEMINI_API_KEY> .env
    echo .env file created. Please edit it to add your Gemini API key.
    exit /b 1
)

REM Install required packages
echo Installing required packages if needed...
pip install llama-index google-generativeai python-dotenv requests pymupdf

REM Run the test
echo Running semantic chunking test on PDF...
python test_pdf_semantic_chunking.py

echo Done! 