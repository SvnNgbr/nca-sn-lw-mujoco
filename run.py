#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
from pathlib import Path

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def run_command(cmd, cwd=None):
    print(f"\n$ {' '.join(cmd)}")
    print("-"*40)
    try:
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        for line in process.stdout:
            print(line, end="")
        process.wait()
        if process.returncode != 0:
            print(f"\nERROR: Command exited with code {process.returncode}")
            return False
        return True
    except Exception as e:
        print(f"\nERROR: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Burpee Training and Video Generation"
    )
    parser.add_argument(
        "--version",
        choices=["v3", "v5"],
        default="v5",
        help="Which version to run? (v3 or v5)"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "video", "both"],
        default="both",
        help="What to execute? (train, video, both)"
    )
    parser.add_argument(
        "--root-assist",
        type=float,
        default=0.0,
        help="Root assist for video generation"
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and use existing model"
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent.absolute()
    print_header(f"BURPEE TRAINING - Version: {args.version.upper()}")

    # Version-specific paths
    if args.version == "v3":
        script_dir = base_dir / "files_imitationv2"
        run_dir = base_dir / "runs_v3_physical"
        train_script = "train_burpee_curriculum.py"
        watch_script = "watch_trained_burpee_v2.py"
        model_prefix = "burpee_ppo_curriculum"
    else:
        script_dir = base_dir / "files_v5"
        run_dir = base_dir / "runs_humanoid_v5"
        train_script = "v5train.py"
        watch_script = "v5watch.py"
        model_prefix = "humanoid_v5_curriculum_final"

    models_dir = run_dir / "models"
    videos_dir = run_dir / "videos"
    logs_dir = run_dir / "logs"

    run_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    videos_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    print(f"Scripts: {script_dir}")
    print(f"Working directory: {run_dir}")

    # Model path
    model_path = models_dir / f"{model_prefix}.zip"

    # Training
    if args.mode in ["train", "both"] and not args.skip_train:
        print_header(f"STARTING TRAINING ({args.version.upper()})")
        env = os.environ.copy()
        env["MODEL_DIR"] = str(models_dir)
        env["LOG_DIR"] = str(logs_dir)

        if not run_command(["python", train_script], cwd=script_dir):
            print("Training failed.")
            sys.exit(1)

        # Move model to models directory
        model_files = list(script_dir.glob("*.zip")) + list(script_dir.glob("trained_models/**/*.zip"))
        for mf in model_files:
            if "burpee" in mf.name or "humanoid" in mf.name:
                dest = models_dir / mf.name
                mf.rename(dest)
                print(f"Model moved: {dest}")

        print_header("TRAINING COMPLETE")
    else:
        print("Training skipped.")

    # Check for model
    if args.mode in ["video", "both"]:
        if not model_path.exists():
            existing = list(models_dir.glob("*.zip"))
            if existing:
                model_path = existing[0]
                print(f"Using: {model_path}")
            else:
                print(f"No model found in {models_dir}.")
                if args.mode == "video":
                    sys.exit(1)

    # Video generation
    if args.mode in ["video", "both"]:
        print_header(f"GENERATING VIDEO ({args.version.upper()})")
        print(f"Model: {model_path}")
        print(f"Root assist: {args.root_assist}")

        # Model path relative to script_dir
        try:
            rel_model_path = model_path.relative_to(script_dir)
        except ValueError:
            rel_model_path = model_path

        video_cmd = [
            "python", watch_script,
            "--model", str(rel_model_path),
            "--root-assist", str(args.root_assist)
        ]
        if not run_command(video_cmd, cwd=script_dir):
            print("Video generation failed.")
            sys.exit(1)

        # Move video to videos directory
        video_files = list(script_dir.glob("*.mp4")) + list(script_dir.glob("*.gif"))
        for vf in video_files:
            dest = videos_dir / vf.name
            vf.rename(dest)
            print(f"Video saved: {dest}")

        print_header("VIDEO GENERATION COMPLETE")

    # Summary
    print_header("SUMMARY")
    print(f"Version: {args.version.upper()}")
    print(f"Mode: {args.mode}")
    print(f"Model: {model_path}")
    if args.mode in ["video", "both"]:
        video_files = list(videos_dir.glob("*.gif")) + list(videos_dir.glob("*.mp4"))
        if video_files:
            print(f"Videos: {len(video_files)} in {videos_dir}")
            for vf in video_files:
                print(f"  - {vf}")
    print("="*60)
    print("DONE!")

if __name__ == "__main__":
    main()