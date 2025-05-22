: # Hybrid script to install requirements and launch KoboldCpp
: # For Users who want to run KoboldCpp unpacked
: # Works on both Windows and Linux
:<<BATCH
    @echo off
    echo This script will setup and launch KoboldCpp, intended for those running it unpacked.
    echo Checking for Python on Windows...
    where python >nul 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo Python is not installed or not in PATH.
        echo Please download it from https://www.python.org/downloads/windows/
        GOTO END
    )

    echo Checking Python dependencies on Windows...
    python -m pip install --upgrade pip
    python -m pip install -r requirements_minimal.txt

    echo Running koboldcpp.py on Windows...
    python koboldcpp.py
    GOTO END
BATCH

echo "This script will setup and launch KoboldCpp, intended for those running it unpacked."
echo "Checking for Python on Linux..."
if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed or not in PATH."
    echo "Please install it using your package manager (e.g., sudo apt install python3)."
    exit 1
fi

echo "Checking Python dependencies on Linux..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements_minimal.txt

echo "Running koboldcpp.py on Linux..."
python3 koboldcpp.py
exit

:END
