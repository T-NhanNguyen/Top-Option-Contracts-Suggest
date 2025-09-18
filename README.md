# Top-Option-Contracts-Suggest
Suggest to you the top option contracts from a given Ticker. Uses Yfinance. Simple Python script

## Windows (Using PowerShell)

Install Python:
    
    Go to python.org/downloads and grab Python 3.12 (or latest stable).
    Run the installer. Important: Check "Add Python to PATH" on the first screen. Install for "Current User" (no admin needed).
    Restart PowerShell after install.


Set Up Project macOS and WSL:
    python3 -m venv venv
    source venv/bin/activate
    pip3 install -r requirements.txt

    python3 main.py AAPL

Set Up Project Windows (Bash):
    python -m venv venv
    .\venv\Scripts\activate
    pip install -r requirements.txt

    python main.py AAPL or python .\main.py AAPL