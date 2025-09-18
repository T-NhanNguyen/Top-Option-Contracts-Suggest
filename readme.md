## Windows (Using PowerShell)
Install Python:

    Go to python.org/downloads and grab Python 3.12 (or latest stable).
    Run the installer. Important: Check "Add Python to PATH" on the first screen. Install for "Current User" (no admin needed).
    Restart PowerShell after install.
    
### Create Virtual Environment & Install Deps:
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Run Application
python main.py AAPL

## macOS (Using Terminal)
Install Python:

    Go to python.org/downloads and grab the macOS installer for Python 3.12 (or latest stable).
    Run the .pkg installer (no admin needed for user install).
    Restart Terminal after.

### Create Virtual Environment & Install Deps:
```
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

### Run Application
python3 main.py AAPL