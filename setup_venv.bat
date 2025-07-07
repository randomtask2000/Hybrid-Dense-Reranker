@echo off
REM Hybrid Dense Reranker - Virtual Environment Setup Script for Windows

echo 🚀 Setting up Hybrid Dense Reranker virtual environment...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️ Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo 📚 Installing dependencies...
pip install -r requirements.txt

REM Copy environment file if it doesn't exist
if not exist .env (
    echo 📝 Creating .env file from template...
    copy .env.example .env
    echo ⚠️  Please edit .env file and add your ANTHROPIC_API_KEY
) else (
    echo ✅ .env file already exists
)

echo.
echo 🎉 Setup complete!
echo.
echo Next steps:
echo 1. Edit .env file and add your ANTHROPIC_API_KEY
echo 2. Activate the virtual environment: venv\Scripts\activate
echo 3. Test the setup: python test_embedding.py
echo 4. Run the application: python app.py
echo.
echo To deactivate the virtual environment later, run: deactivate
pause