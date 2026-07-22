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
        choices=["train", "video", "plots", "both", "all"],
        default="all",
        help="What to execute? (train, video, plots, both, all)"
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
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom name for this run (used for plots and directories)"
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent.absolute()
    
    if args.run_name:
        run_name = args.run_name
    else:
        import time
        run_name = time.strftime("%Y%m%d_%H%M%S")

    print_header(f"BURPEE TRAINING - Version: {args.version.upper()} - Run: {run_name}")

    if args.version == "v3":
        script_dir = base_dir / "files_imitation"
        train_script = "train_burpee_v3_physical.py"
        watch_script = "watch_trained_burpee_v3_physical.py"
        video_script = "burpee_robot_v3_parallel_arms.py"
        tensorprint_script = "tensorprint.py"
        model_prefix = "burpee_v3_physical"
        run_dir = script_dir / "runs_v3_physical"
        models_dir = script_dir / "trained_models"
        videos_dir = script_dir / "videos"
        plots_dir = script_dir / "training_plots"
        logs_dir = run_dir
    else:
        script_dir = base_dir / "files_v5"
        train_script = "v5train.py"
        watch_script = "v5watch.py"
        video_script = "v5vid.py"
        tensorprint_script = "v5tensorprint.py"
        model_prefix = "humanoid_v5_curriculum_final"
        run_dir = script_dir / "runs_humanoid_v5"
        models_dir = script_dir / "trained_models_humanoid_v5"
        videos_dir = script_dir / "videos"
        plots_dir = script_dir / "training_plots"
        logs_dir = run_dir

    run_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    videos_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    print(f"Scripts: {script_dir}")
    print(f"Working directory: {run_dir}")
    print(f"Models: {models_dir}")
    print(f"Videos: {videos_dir}")
    print(f"Plots: {plots_dir}")

    model_path = models_dir / f"{model_prefix}.zip"

    # ============================================================
    # 1. TRAINING
    # ============================================================
    if args.mode in ["train", "both", "all"] and not args.skip_train:
        print_header(f"STARTING TRAINING ({args.version.upper()})")

        if not run_command(["python", train_script], cwd=script_dir):
            print("Training failed.")
            sys.exit(1)

        print_header("TRAINING COMPLETE")
        
        if not model_path.exists():
            existing = list(models_dir.glob("*.zip"))
            if existing:
                model_path = existing[0]
                print(f"Model found: {model_path}")
    else:
        print("Training skipped.")

    # ============================================================
    # 2. MODELL PRÜFEN
    # ============================================================
    if args.mode in ["video", "both", "all"]:
        if not model_path.exists():
            existing = list(models_dir.glob("*.zip"))
            if existing:
                model_path = existing[0]
                print(f"Using: {model_path}")
            else:
                print(f"No model found in {models_dir}.")
                if args.mode == "video":
                    sys.exit(1)

    # ============================================================
    # 3. VIDEO GENERIEREN
    # ============================================================
    if args.mode in ["video", "both", "all"]:
        print_header(f"GENERATING VIDEO ({args.version.upper()})")
        print(f"Model: {model_path}")
        print(f"Root assist: {args.root_assist}")

        video_cmd = [
            "python", video_script,
            "--model", str(model_path),
            "--root-assist", str(args.root_assist)
        ]
        if not run_command(video_cmd, cwd=script_dir):
            print("Video generation failed.")
            sys.exit(1)

        # Video in videos_dir verschieben
        video_files = list(script_dir.glob("*.mp4")) + list(script_dir.glob("*.gif"))
        for vf in video_files:
            dest = videos_dir / vf.name
            if dest.exists():
                dest.unlink()
            vf.rename(dest)
            print(f"Video saved: {dest}")

        print_header("VIDEO GENERATION COMPLETE")

    # ============================================================
    # 4. PLOTS GENERIEREN
    # ============================================================
    if args.mode in ["plots", "both", "all"]:
        print_header(f"GENERATING TRAINING PLOTS ({args.version.upper()})")
        
        if logs_dir.exists():
            subdirs = [d for d in logs_dir.iterdir() if d.is_dir() and d.name.startswith("PPO_")]
            if subdirs:
                actual_logdir = max(subdirs, key=lambda d: d.stat().st_mtime)
                print(f"Using TensorBoard logs from: {actual_logdir}")
                
                plot_cmd = [
                    "python", str(script_dir / tensorprint_script),
                    "--logdir", str(actual_logdir),
                    "--run-name", f"{args.version}_{run_name}",
                    "--output-dir", str(plots_dir),
                    "--window", "20"
                ]
                if not run_command(plot_cmd, cwd=base_dir):
                    print("Plot generation failed.")
                else:
                    print_header("PLOTS GENERATED")
                    print(f"Plots saved to: {plots_dir}")
            else:
                print(f"No PPO subdirectories found in {logs_dir}")
        else:
            print(f"TensorBoard log directory not found: {logs_dir}")

    # ============================================================
    # 5. SUMMARY
    # ============================================================
    print_header("SUMMARY")
    print(f"Version: {args.version.upper()}")
    print(f"Run name: {run_name}")
    print(f"Mode: {args.mode}")
    print(f"Model: {model_path}")
    print(f"All results are in: {script_dir}")
    
    video_files = list(videos_dir.glob("*.gif")) + list(videos_dir.glob("*.mp4"))
    if video_files:
        print(f"Videos: {len(video_files)} in {videos_dir}")
        for vf in video_files:
            print(f"  - {vf}")
    
    plot_files = list(plots_dir.glob("*.png")) + list(plots_dir.glob("*.pdf"))
    if plot_files:
        print(f"Plots: {len(plot_files)} in {plots_dir}")
        for pf in plot_files:
            print(f"  - {pf}")
    
    model_files = list(models_dir.glob("*.zip"))
    if model_files:
        print(f"Models: {len(model_files)} in {models_dir}")
        for mf in model_files:
            print(f"  - {mf}")
    
    print("="*60)
    print("DONE!")

if __name__ == "__main__":
    main()