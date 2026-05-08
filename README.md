## ⚙️ Environment Setup

This project uses Python 3.10. We recommend using [Conda](https://docs.conda.io/en/latest/miniconda.html) to manage your virtual environment. Create a new environment named (`ml_opt` by default)  with Python 3.10 and `pip` and install the dependencies:
```bash
conda create --name ml_opt python=3.10 pip -y
conda activate ml_opt
pip install -r requirements.txt
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```
