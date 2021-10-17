<div align="center">
<h1>Setup Guide Rep</a></h1>
by Hongnan Gao
<br>
</div>

---

- [Introduction](#introduction)
- [Step-by-Step Guide](#step-by-step-guide)
  - [Setting up virtual env and requirements](#setting-up-virtual-env-and-requirements)
  - [Git:](#git)
  - [Command Line](#command-line)
  - [Documentation](#documentation)
    - [Type Hints](#type-hints)
    - [Mkdocs + Docstrings](#mkdocs--docstrings)
  - [Misc Problems](#misc-problems)

---

# Introduction

This guide serves as an end-to-end cycle for creating scripts for Windows/Ubuntu/Mac system.

Please note that you should set up both **Git**, **Virtual Environment** and **Cuda** in your respective systems. You can refer the below two links for installation on Ubuntu System. 

- [Cuda Installation for Ubuntu](https://www.cyberciti.biz/faq/ubuntu-linux-install-nvidia-driver-latest-proprietary-driver/)
- [Git Installation for Ubuntu](https://wiki.paparazziuav.org/wiki/Github_manual_for_Ubuntu)

---

# Step-by-Step Guide

## Open VSCode

Assuming VSCode setup, open up terminal/powershell in your respective system and type:
    ```bash
    code "path to folder"
    ```
    to open up VSCode.

## Virtual Environment

In VSCode, you want to set up a virtual environtment, make sure virtual environment is installed in your system, for example, in Ubuntu, you can call:

```bash
# For Ubuntu
sudo apt install python3.8 python3.8-venv python3-venv
```

```bash
# For Mac
pip3 install virtualenv
```

Subsequently, you can activate the VM as follows:

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

```bash
# Assuming Mac
virtualenv venv_bcw
source venv_bcw/bin/activate
```

Note if you encounter some weird issues in windows, you can check if it is the admin bug [here](https://stackoverflow.com/questions/54776324/powershell-bug-execution-of-scripts-is-disabled-on-this-system)

Finally, in your activate virtual environment, we will upgrade pip.
```bash
# upgrade pip
python -m pip install --upgrade pip setuptools wheel
```

---

## Setup and requirements

Create a file named `setup.py` and `requirements.txt` concurrently. The latter should have the libraries that one is interested in having for his project while the formal is a `setup.py` file where it contains the setup object which describes how to set up our package and it's dependencies. The first several lines cover metadata (name, description, etc.) and then we define the requirements. Here we're stating that we require a Python version equal to or above 3.8 and then passing in our required packages to install_requires. Finally, we define extra requirements that different types of users may require. This is a standard practice and more can be understood from [madewithml](madewithml.com).

The user can now call the following commands to install the dependencies in their own virtual environment.

```bash
pip install -e .  # installs required packages only       
python -m pip install -e ".[dev]"                                       # installs required + dev packages
python -m pip install -e ".[test]"                                      # installs required + test packages
python -m pip install -e ".[docs_packages]"                             # installs required documentation packages
```

> Something worth taking note is when you download PyTorch Library, there is a dependency link since we are downloading cuda directly, you may execute as such:
```bash
pip install -e . -f https://download.pytorch.org/whl/torch_stable.html
```

> For developers, you may also need to use `test_packages, dev_packages and docs_packages` as well.

> Entry Points: In the final lines of the `setup.py` we defined various entry points we can use to interact with the application. Here we define some console scripts (commands) we can type on our terminal to execute certain actions. For example, after we install our package, we can type the command `reighns_bcw ` to run the app variable inside cli.py.

---

## Git
 
### Normal workflow

1. In the VSCode environment, type `git init` to initialize the folder as a local repository.
2. Create `.gitignore` to put files that you do not want `git` to commit. This is important as you do not want to commit very large datasets.
3. After you have created new files, use `git add .` to add all existing files. If you want to add individual file just type `git add "filename.py"`.
4. You can call `git status` to check status of tracked vs untracked files.
5. Finally, if you are ready to commit the changes you've made, type `git commit -a` and write your message, or `git commit -a main.py`. Note that if the editor does not pop up and instead ask you to commit the message in the terminal, you can type `git config --global core.editor "code --wait"` to ask git to show you an proper editor. 
6. Note everything done above is still at local repository, as a result, we now need to push the changes into GitHub's remote server by calling 
    ```bash
    # add remote origin
    git remote add origin "your github path"
    # remove remote origin if needed
    git remove origin "your github path"
    ```
    so that you have added this local repo to the GitHub. 

8. Finally, we use `git push origin master` to push to master branch.

9. Now step 8 may fail sometimes if you did not verify credentials, a fix is instead of step 7 and 8, we replace with 
    ```bash
    git remote add origin your-repo-http
    git remote set-url origin your-repo-http
    git push origin master
    ```

---

1. One thing worth noting is that if you created the repository with some files in it, then the above steps will lead to error.
2. Instead, to overwrite : do `git push -f origin master` . For newbies, it is safe to just init a repo with no files inside in the remote to avoid the error "Updates were rejected because the tip..."


### Pulling

If you are writing code on your main desktop, say system A and committing a lot of changes, and when you are outside you are using your portable macbook, say system B, you may want to do the following:

1. If your last commit of system A is at commit 6, and system B is at commit 3, where commit 4-6 were done on A, then you can simply call:

    ```bash
    git config --global pull.rebase false  # merge (the default strategy)
    git pull 
    ```

2. However, if you have uncommited changes on system B and did not commit yet, then what you can do is call:

    ```bash
    git stash # this saves a snapshot of your current work and remove the commit
    git pull
    ```

---

### Useful Git Commands


---

## Command Line

Something worth noting is we need to use dash instead of underscore when calling a function in command line.

reighns_linear_regression regression-test --solver "Batch Gradient Descent" --num-epochs 500

## Documentation

### Type Hints

### Mkdocs + Docstrings

1. Copy paste the template from Goku in, most of what he use will be in `mkdocs.yml` file. Remember to create the `mkdocs.yml` in the root path.
2. Then change accordingly the content inside `mkdocs.yml`, you can see my template that I have done up.
3. Remember to run `python -m pip install -e ".[docs_packages]" ` to make sure you do have the packages.
4. Along the way you need to create a few folders, follow the page tree in mkdocs.yml, everything should be created in `docs/` folder.
5. As an example, in our reighns-linear-regression folder, we want to show two scenarios:
    - Scenario 1: I merely want a full markdown file to show on the website. In this case, in the "watch", we specify a path we want to watch in our `docs/` folder. In this case, I created a `documentation` folder under `docs/` so I specify that. Next in the `docs/documentation/` folder I create a file called `linear_regression.md` where I dump all my markdown notes inside. Then in the `nav` tree in `mkdocs.yml`, specify
    ```
    nav:
    - Home:
        - Introduction: index.md
    - Getting started: getting_started.md
    - Detailed Notes:
        - Notes: documentation/linear_regression.md
        - Reference: documentation/reference_links.md
    ```
    Note that Home and Getting Started are optional but let us keep it for the sake of completeness. What you need to care is "Detailed Notes" and note the path I gave to them - which will point to the folders in `docs/documentation/`.

    - Scenario 2: I want a python file with detailed docstrings to appear in my static website. This is slightly more complicated. First if you want a new section of this you can create a section called `code_reference`, both under the `nav` above and also in the folder `docs/`, meaning `docs/code_reference/` must exist. Put it under watch as well. Now in `docs/code_reference/` create a file named say `linear_regression_from_scratch.md` and put `::: src.linear_regression` inside, note that you should not have space in between.
    
    
---


## Misc Problems

- How to import modules with dash

https://stackoverflow.com/questions/65861999/how-to-import-a-module-that-has-a-dash-in-its-name-without-renaming-it

- How to show nbextensions properly https://stackoverflow.com/questions/49647705/jupyter-nbextensions-does-not-appear/50663099

