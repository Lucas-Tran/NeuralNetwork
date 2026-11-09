# NeuralNetwork

## Local Setup

Requirements: Python 3.10 or newer. The setup script selects the first available
interpreter from Python 3.12, 3.11, 3.10, then `python3`.

From the repository root, create the virtual environment and install dependencies:

```bash
./setup.sh
```

The setup script creates a root-level `activate` symlink to the virtual
environment. Activate it in your terminal, then launch the app:

```bash
source ./activate
python main.py
```

Open [http://localhost:7860](http://localhost:7860) to use the drawing interface.
Keep the terminal open while the app is running. To use a specific compatible
Python interpreter, set `PYTHON`, for example `PYTHON=python3.12 ./setup.sh`.

If `.venv` already exists, make sure it uses Python 3.10 or newer before running
the setup script; creating a virtual environment in an existing directory does
not change that environment's Python version.
