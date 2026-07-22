import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator

def find_event_files(logdir):
    event_files = []
    for root, dirs, files in os.walk(logdir):
        for f in files:
            if f.startswith("events.out.tfevents"):
                event_files.append(os.path.join(root, f))
    return event_files

def extract_tb_data(event_file, tags):
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    
    print("\nAvailable scalar tags:")
    if 'scalars' in ea.Tags():
        for tag in sorted(ea.Tags()['scalars']):
            print(f"  {tag}")
    print()
    
    data = {}
    for tag in tags:
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            if events:
                data[tag] = {
                    "steps": np.array([e.step for e in events]),
                    "values": np.array([e.value for e in events])
                }
                print(f"Loaded {tag}: {len(events)} points")
        else:
            print(f"Tag not found: {tag}")
    
    return data

def smooth_data(values, window=20):
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window)/window, mode='valid')

def plot_metrics(data, output_dir, run_name, window=20):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Training Metrics - {run_name}", fontsize=16, fontweight="bold")
    
    # Definiere die 4 Metriken, die geplottet werden sollen
    metrics = [
        ("rollout/ep_rew_mean", "Episode Reward", "Reward", "#1f77b4"),
        ("rollout/ep_len_mean", "Episode Length", "Steps", "#ff7f0e"),
        ("train/learning_rate", "Learning Rate", "Rate", "#2ca02c"),
        ("time/fps", "Frames per Second (FPS)", "FPS", "#d62728"),
    ]
    
    plot_idx = 0
    for tag, title, ylabel, color in metrics:
        row = plot_idx // 2
        col = plot_idx % 2
        ax = axes[row, col]
        
        # Versuche alternative Tag-Namen
        found = False
        for alt_tag in [tag, tag.split('/')[-1]]:
            if alt_tag in data and len(data[alt_tag]["steps"]) > 0:
                d = data[alt_tag]
                found = True
                break
        
        if not found:
            ax.text(0.5, 0.5, f"No data for {title}", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title(title)
            ax.set_visible(True)
            plot_idx += 1
            continue
        
        steps = d["steps"]
        values = d["values"]
        
        # Raw data
        ax.plot(steps, values, linewidth=1, color=color, alpha=0.4, label="Raw")
        
        # Smoothed data
        if len(values) > window:
            smoothed = smooth_data(values, window)
            smooth_steps = steps[:len(smoothed)]
            ax.plot(smooth_steps, smoothed, linewidth=2, color=color, 
                   label=f"Smoothed (n={window})")
        
        ax.set_xlabel("Timesteps", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plot_idx += 1
    
    plt.tight_layout()
    
    png_path = output_dir / f"{run_name}_metrics.png"
    pdf_path = output_dir / f"{run_name}_metrics.pdf"
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    
    return png_path, pdf_path

def generate_summary(data, output_dir, run_name):
    summary_path = output_dir / f"{run_name}_summary.txt"
    
    with open(summary_path, "w") as f:
        f.write(f"Training Summary - {run_name}\n")
        f.write("="*60 + "\n\n")
        
        for tag, d in data.items():
            if len(d["values"]) > 0:
                final_val = d["values"][-1]
                mean_val = np.mean(d["values"])
                max_val = np.max(d["values"])
                min_val = np.min(d["values"])
                
                f.write(f"{tag}:\n")
                f.write(f"  Final: {final_val:.4f}\n")
                f.write(f"  Mean:  {mean_val:.4f}\n")
                f.write(f"  Max:   {max_val:.4f}\n")
                f.write(f"  Min:   {min_val:.4f}\n")
                f.write("\n")
        
        f.write("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Generate training plots and summaries from TensorBoard logs."
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="runs_humanoid_v5",
        help="Directory containing TensorBoard logs"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for the run (used in output filenames)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="training_plots",
        help="Directory to save plots and summaries"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="Window size for smoothing"
    )
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent.absolute()
    logdir = Path(args.logdir)
    if not logdir.is_absolute():
        logdir = base_dir / logdir
    
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(exist_ok=True)
    
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = logdir.name
    
    print(f"Processing logs from: {logdir}")
    print(f"Output directory: {output_dir}")
    print(f"Run name: {run_name}")
    
    event_files = find_event_files(logdir)
    if not event_files:
        print(f"No event files found in {logdir}")
        return
    
    latest_event = max(event_files, key=os.path.getmtime)
    print(f"Using event file: {latest_event}")
    
    all_tags = [
        "rollout/ep_rew_mean",
        "rollout/ep_len_mean",
        "train/learning_rate",
        "time/fps",
        "ep_rew_mean",
        "ep_len_mean",
        "learning_rate",
        "fps",
    ]
    
    data = extract_tb_data(latest_event, all_tags)
    
    if not data:
        print("No data extracted. Check that the event file contains scalar data.")
        return
    
    png_path, pdf_path = plot_metrics(data, output_dir, run_name, args.window)
    print(f"\nPlots saved to:")
    print(f"  PNG: {png_path}")
    print(f"  PDF: {pdf_path}")
    
    generate_summary(data, output_dir, run_name)
    print(f"Summary saved to: {output_dir / f'{run_name}_summary.txt'}")
    
    for tag, d in data.items():
        if len(d["steps"]) > 0:
            csv_path = output_dir / f"{run_name}_{tag.replace('/', '_')}.csv"
            np.savetxt(csv_path, np.column_stack([d["steps"], d["values"]]), 
                      delimiter=",", header="steps,values", comments="")
    print(f"Raw data saved as CSV files in: {output_dir}")

if __name__ == "__main__":
    main()