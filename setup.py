import subprocess
import os
import sys

def ask_user_for_confirmation(prompt: str) -> bool:
    while True:
        user_input = input(prompt).strip().lower()
        if user_input == "y":
            return True
        elif user_input == "n":
            return False
        else:
            print("Invalid input. Please use only 'y' or 'n'.")

def in_conda() -> bool:
    # Prüfe, ob die Umgebungsvariable CONDA_DEFAULT_ENV existiert (Conda/Miniforge)
    return "CONDA_DEFAULT_ENV" in os.environ

def in_venv() -> bool:
    # Prüfe, ob das Skript in einer virtuellen Umgebung läuft
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

if __name__ == "__main__":
    if in_conda():
        print("Detected conda environment:", os.environ.get("CONDA_DEFAULT_ENV"))
    elif in_venv():
        print("Detected virtual environment:", sys.prefix)
    else:
        print("No conda or virtual environment detected. Aborting...")
        raise EnvironmentError()

    print("Starting Setup...")

    mujocopath = os.path.join(
        os.path.dirname(__file__),
        "mujoco"
    )

    if not os.path.exists(mujocopath):
        print("Cloning the MuJoCo repository...")
        subprocess.check_call(
            [
                "git",
                "clone",
                "https://github.com/deepmind/mujoco.git"
            ],
            cwd=os.path.dirname(__file__)
        )
        print("Done!")
    else:
        print("MuJoCo repository already cloned or folder exists. Skipping...")

    # Dummy variable for init reasons
    blackwell = False
    print("Installing dependencies...")
    if ask_user_for_confirmation("Blackwell GPU? (y/n): "):
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-r",
                os.path.join(
                    os.path.dirname(__file__),
                    "requirements_blackwell.txt"
                )
            ]
        )
        blackwell = True
    else:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-r",
                os.path.join(
                    os.path.dirname(__file__),
                    "requirements.txt"
                )
            ]
        )
    print("Done!")

    if blackwell:
        print("Torch installation for Blackwell-series GPUs not possible in script, please refer to: [link to documentation].")
        print("Probably like this: 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129'")

    print("Setup complete!")