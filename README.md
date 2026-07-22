# nca-sn-lw-mujoco

Neuromorphic Control Assignment für den SoSe Kurs von Dr. rer. nat. Johannes Maria Leugering for Introduction to Neuromorphic Control

We used the mujoco repository at <https://mujoco.org> and <https://github.com/google-deepmind/mujoco/tree/main> to build upon it.
Also we used Humanoid V5 (only XML) <https://gymnasium.farama.org/environments/mujoco/humanoid/>

Link to Presentation:
https://www.figma.com/board/lRynw06d1QNUb4VMZNYzNT/NC-Assignment?node-id=0-1&t=1X3BSlGuk0xQx8nC-1

Link to Report:
https://docs.google.com/document/d/1L832A2-83RfPdtDlJrWloBiF8Uhi3MdTyWWzJrtdEYY/edit?usp=sharing

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

# How to run:

 V5 training + video (standard)
```bash
python run.py
```
only V5 training
```bash
python run.py --mode train
```
only V5 video with prev. existing modell
```bash
python run.py --mode video --skip-train
```
V3 training + video
```bash
python run.py --version v3
```
V3 video with root-Assist
```bash
python run.py --version v3 --mode video --skip-train --root-assist 0.5
```
V5 video with root-Assist
```bash
python run.py --mode video --skip-train --root-assist 0.5
```
