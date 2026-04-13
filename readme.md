
## Installation

### 1. Clone the repository

```bash
git clone git@github.com:1511878618/EpiDL.git
cd EpiDL
```

### 2. Create a conda environment

```bash
conda create -n epidl python=3.12
conda activate epidl
```

### 3. Install dependencies

```bash
pip install epilearn
pip install Eir
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0
pip install torch_geometric
pip install epydemix
```

### 4. Install EpiDL in editable mode

```bash
pip install -e .
```

