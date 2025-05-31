#!/bin/bash

# Exit on any error
set -e

if [ "$(uname -o)" != "Android" ]; then
echo "Error: This script is only intended for Termux on Android!"
exit 1
fi

echo "--------------------------------------------"
echo "Esobold Quick Installer for Termux (Android only!)"
echo "--------------------------------------------"
if [ $# -ge 1 ]; then
    choice="$1"
    echo "Using command-line argument: $choice"
# Check if running interactively (terminal input)
elif [ -t 0 ]; then
    # Running interactively
    echo "[1] - Proceed to install and launch with default options - no model is loaded by default, but can be downloaded in the UI"
    echo "[2] - Proceed to install and not run."
    echo "[3] - Exit script"
    echo "--------------------------------------------"
    read -p "Enter your choice [1-3]: " choice
else
    # Non-interactive, default to choice 1
    echo "Defaulting to normal install and model download. Run script interactively for other options. Install will start in 3 seconds."
    choice="1"
    sleep 3
fi

if [ "$choice" = "3" ]; then
    echo "Exiting script. Goodbye!"
    exit 0
elif [ "$choice" = "2" ]; then
    echo "[*] Install without model download..."
    INSTALL_MODEL=false
elif [ "$choice" = "1" ]; then
    echo "[*] Install with model downloader enabled in Lite UI..."
    INSTALL_MODEL=true
else
    echo "Invalid choice. Exiting."
    exit 1
fi

echo "[*] Checking Dependencies..."
check_wget=$(command -v wget || true)
check_git=$(command -v git || true)
check_python=$(command -v python || true)
if [ -n "$check_wget" ] && [ -n "$check_git" ] && [ -n "$check_python" ]; then
    echo "[*] Dependencies are already installed..."
else
    echo "[*] Setup dependencies..."
    apt update
    DEBIAN_FRONTEND=noninteractive apt-get install -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" openssl
    pkg install -y wget git python
    pkg upgrade -o Dpkg::Options::="--force-confold" -y
fi

# Determine script directory (works for both curl|sh and ./install.sh)
if [ -f "$0" ]; then
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"  # Normal execution (./install.sh)
else
    SCRIPT_DIR="$(pwd)"  # Piped execution (curl | sh)
fi
# Check if koboldcpp.py already exists nearby
if [ -f "$SCRIPT_DIR/koboldcpp.py" ]; then
    echo "[*] Detected existing koboldcpp.py in $SCRIPT_DIR"
    KOBOLDCPP_DIR="$SCRIPT_DIR"
elif [ -d "$SCRIPT_DIR/esobold" ] && [ -f "$SCRIPT_DIR/esobold/koboldcpp.py" ]; then
    echo "[*] Detected existing Esobold clone in $SCRIPT_DIR/esobold"
    KOBOLDCPP_DIR="$SCRIPT_DIR/esobold"
else
    echo "[*] No existing esobold found. Cloning repository..."
    cd "$SCRIPT_DIR"
    git clone https://github.com/esolithe/esobold.git
    KOBOLDCPP_DIR="$SCRIPT_DIR/esobold"
fi

# build if needed
cd "$KOBOLDCPP_DIR"
if [ -f "$KOBOLDCPP_DIR/koboldcpp_default.so" ]; then
    echo "[*] Found koboldcpp_default.so — skipping build step."
else
    echo "[*] Building Esobold now..."
    make -j 2
fi

if [[ ! -e "$SCRIPT_DIR/Default.kcpps" ]]; then
    echo "[*] Adding default config"
    echo -e '{}' >> Default.kcpps
fi

# grab model if needed
echo "[*] Your Esobold Installation is Complete!"
if [ "$INSTALL_MODEL" = true ]; then
    echo "[*] Running Esobold..."
    # python koboldcpp.py --model https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf
    python koboldcpp.py --nomodel --adminallowhf --maxrequestsize 100 --admin --admindir "./" --admintextmodelsdir "./" --admindatadir "./" --savedatafile "./"
else
    echo "To use it, please obtain a GGUF model, then run it with the command 'python koboldcpp.py --model (your_gguf)' and then open a web browser to http://localhost:5001"
fi
