# Online Learning Application - Development Setup

## Setup Instructions

### Non-Nix users:
```bash
# Clone the repository
git clone <repo-url>
cd Online-Learning-Application-project

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Register Jupyter kernel
python -m ipykernel install --user --name "$(basename $PWD)-venv"
```

## Adding New Dependencies


1. Install the package with `pip install <package-name>`
1. Add the package to `requirements.txt` with `pip freeze > requirements.txt`
2. For Nix users: run `direnv reload`
3. For non-Nix users: run `pip install -r requirements.txt`

### Kernel not appearing:
```bash
# Re-register the kernel
python -m ipykernel install --user --name "$(basename $PWD)-venv" --force
```