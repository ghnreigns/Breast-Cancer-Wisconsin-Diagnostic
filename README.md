## Directory Structure

## Workflows

---

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

3. Get data
```bash
# Download to data/
tagifai download-data

# or Pull from DVC
dvc init
dvc remote add -d storage stores/blob
dvc pull
```

3. Compute features
```bash
tagifai compute-features
```