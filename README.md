<div align="center">
<h1>Technical Report: Breast Cancer Wisconsin</a></h1>
by Hongnan Gao
<br>
</div>

## Introduction

This repository describes in detail a project on Breast Cancer detection. It holds both step by step walkthrough in notebook format, which I have converted into pdf for viewers, and also source codes in `.py` files.

## Directory Structure

```bash
app/
├── cli.py              - serves as the main driver file to run
config/
├── global_params.py    - global parameters and configurations
data/
├── images              - saved plots
├── processed           - processed data
├── raw                 - raw data
notebooks/              - contain detailed description of each stage of the ML pipeline
src
├── clean.py            - preliminary data cleaning 
├── eval.py             - evaluation of models
├── make_folds.py       - make cross-validation folds
├── plots.py            - plotting functions
├── train.py            - training/optimization pipelines
└── utils.py            - utility functions
```
---

## Workflows

1. Set up virtual environment. Here I gave two versions using Windows or Linux.
   
```bash
# Assuming Windows
python -m venv venv_bcw
.\venv_bcw\Scripts\activate
python -m pip install --upgrade pip setuptools wheel # upgrade pip
```

```bash
# Assuming Linux
python3 -m venv venv_bcw
source venv_bcw/bin/activate
python -m pip install --upgrade pip setuptools wheel # upgrade pip
```

---

2. Set up requirements from `setup.py`.

```bash
python -m pip install -e . --no-cache-dir
```

3. Run training code.
```bash
python3 .\app\cli.py
```