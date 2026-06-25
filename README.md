# nca-sn-lw-mujoco
Neuromorphic Control Assignment für den SoSe Kurs von Dr. rer. nat. Johannes Maria Leugering for Introduction to Neuromorphic Control

We used the mujoco repository at LINK and build upon it.

---

# Getting Started
Clone the Repository

```bash
git clone https://github.com/SvnNgbr/nca-sn-lw-mujoco
```

Set up and activate a python environment

```bash
conda create -n mujoco_3_11_15 python=3.11.15
conda activate mujoco_3_11_15
```
**--- OR ---** 
```bash
python -m venv /path/to/new/virtual/environment
source /path/to/new/virtual/environment/bin/activate
```
Navigate to the repository folder
```bash
cd nca-sn-lw-mujoco
```

And execute setup script.
This will install dependencies, clone the Mujoco repository inside the nca-sn-lw-mujoco repository and modify and copy files.
Executing this is **neccessary** for the main script to work.
```bash
python setup.py
```

If you have a **blackwell-series GPU**, which is not yet supported by the stable torch, you might need to install the [nightly torch version](https://pytorch.org/get-started/locally/). 
The setup script informs you in the last couple of lines if that is applicable for you.
We used the following command:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
```

You should now be all set to execute the main programs for each task!

---
