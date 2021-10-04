<div align="center">
<h1>Technical Report: Breast Cancer Wisconsin</a></h1>
by Hongnan Gao
<br>
</div>

## Introduction

This repository describes in detail a project on Breast Cancer detection. It holds both step by step walkthrough in notebook format, which I have converted into pdf for viewers, and also source codes in `.py` files.

The reports can be found in the `reports` folder.

## Disclaimer

1. The assumption of this report is that it is more inclined towards the technical audience. 
2. For a wholesome step by step approach, please do check the 5 pdf files (Stage 0 to Stage 4). They are converted from Jupyter Notebooks for your reference (source `.ipynb` is in `notebooks` folder).
3. LaTeX equations are written inline instead of the neater displayed version to save space. Some mathematical notations are explained in more details in **Appendix**.

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
reports/                - contain all my reports
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

## Timeline

Here is my timeline for this project:

- **Day 1:** Clarifying requirements and sourcing for a dataset. Find a suitable narrative and use case for the problem.
- **Day 2:** Perform Preliminary Inspection of Data and EDA.
- **Day 3:** Derive insights and actions from EDA, for example, multicollinearity seems to be a problem, and we should handle it in `Pipeline`.
- **Day 4:** Coded up a basic pipeline to do spot checking on various classifier for baselines.
- **Day 5:** Set up Hyperparameter Tuning pipeline.
- **Day 6:** Analyze the results, such as feature importance and metric scores.
- **Day 7:** Trying to condense all the steps into 2-page report.

---

## Remarks

Thank you for giving me this opportunity to do this take home project. I look forward to hearing your feedbacks and  suggestions for areas of improvements. 