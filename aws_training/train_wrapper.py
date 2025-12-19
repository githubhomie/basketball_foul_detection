#!/usr/bin/env python3
"""
Training wrapper for AWS with Weights & Biases integration.

This script wraps train_e2e.py to add:
- W&B experiment tracking with live metrics
- Automatic checkpoint backup to S3
- Graceful handling of spot instance interruptions
- YAML-based configuration

Usage:
    python aws_training/train_wrapper.py --config aws_training/configs/v2_baseline.yaml
    python aws_training/train_wrapper.py --config aws_training/configs/v2_baseline.yaml --resume
    python aws_training/train_wrapper.py --config aws_training/configs/v2_baseline.yaml --no-wandb
"""

import os
import sys
import argparse
import subprocess
import signal
import time
import json
import threading
from datetime import datetime
from pathlib import Path

import yaml

# Try to import wandb (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Run: pip install wandb")

# Try to import boto3 for S3 backup (optional)
try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def setup_wandb(config: dict, run_name: str, resume: bool = False):
    """Initialize W&B run with configuration."""
    if not WANDB_AVAILABLE:
        return None

    return wandb.init(
        project="nba-foul-detection",
        name=run_name,
        config=config,
        tags=["aws", config.get("model_arch", "unknown")],
        resume="allow" if resume else None,
        notes=config.get("notes", "")
    )


def sync_to_s3(local_path: str, s3_bucket: str, s3_key: str):
    """Upload a file to S3."""
    if not BOTO3_AVAILABLE:
        return False

    try:
        s3 = boto3.client('s3')
        s3.upload_file(local_path, s3_bucket, s3_key)
        return True
    except Exception as e:
        print(f"S3 upload failed: {e}")
        return False


def build_train_command(config: dict, save_dir: str, resume: bool = False) -> list:
    """Build command-line arguments for train_e2e.py."""

    # Find train_e2e.py (in repo root, same level as aws_training/)
    repo_root = Path(__file__).parent.parent
    train_script = repo_root / "train_e2e.py"

    if not train_script.exists():
        raise FileNotFoundError(f"Could not find train_e2e.py at {train_script}")

    cmd = [
        sys.executable,
        str(train_script),
        config["dataset"],
        config["frame_dir"],
        "-m", config["model_arch"],
        "-t", config["temporal_arch"],
        "--clip_len", str(config["clip_len"]),
        "--batch_size", str(config["batch_size"]),
        "--num_epochs", str(config["num_epochs"]),
        "-lr", str(config["learning_rate"]),
        "-s", save_dir,
        "--crop_dim", str(config.get("crop_dim", 224)),
        "--warm_up_epochs", str(config.get("warm_up_epochs", 3)),
    ]

    # Optional parameters
    if "dilate_len" in config:
        cmd.extend(["--dilate_len", str(config["dilate_len"])])

    if "fg_upsample" in config:
        cmd.extend(["--fg_upsample", str(config["fg_upsample"])])

    if config.get("mixup", False):
        cmd.extend(["--mixup", "true"])

    if "start_val_epoch" in config:
        cmd.extend(["--start_val_epoch", str(config["start_val_epoch"])])

    if resume:
        cmd.append("--resume")

    return cmd


class TrainingMonitor:
    """Monitor training progress and log to W&B."""

    def __init__(self, save_dir: str, wandb_run, s3_bucket: str = None):
        self.save_dir = Path(save_dir)
        self.wandb_run = wandb_run
        self.s3_bucket = s3_bucket
        self.last_epoch = -1
        self.experiment_name = self.save_dir.name
        self.running = True

    def start_monitoring(self):
        """Start background thread to monitor loss.json."""
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop_monitoring(self):
        """Stop the monitoring thread."""
        self.running = False

    def _monitor_loop(self):
        """Background loop to check for new epochs."""
        while self.running:
            self.check_progress()
            time.sleep(30)  # Check every 30 seconds

    def check_progress(self):
        """Check for new checkpoints and log to W&B."""
        loss_file = self.save_dir / "loss.json"
        if not loss_file.exists():
            return

        try:
            with open(loss_file) as f:
                content = f.read().strip()
                # Handle potential incomplete JSON
                if not content.endswith(']'):
                    content = content.rstrip(',') + ']'
                if not content.startswith('['):
                    content = '[' + content
                losses = json.loads(content)
        except (json.JSONDecodeError, Exception):
            return

        for entry in losses:
            epoch = entry.get("epoch", -1)
            if epoch > self.last_epoch:
                self.last_epoch = epoch

                # Log to W&B
                if self.wandb_run:
                    log_data = {
                        "epoch": epoch,
                        "train_loss": entry.get("train", 0),
                        "val_loss": entry.get("val", 0),
                    }

                    # Add mAP if available
                    if "val_mAP" in entry:
                        log_data["val_mAP"] = entry["val_mAP"]
                    if "mAP" in entry:
                        log_data["val_mAP"] = entry["mAP"]

                    wandb.log(log_data)

                # Sync checkpoint to S3
                if self.s3_bucket:
                    ckpt_file = self.save_dir / f"checkpoint_{epoch:03d}.pt"
                    if ckpt_file.exists():
                        s3_key = f"checkpoints/{self.experiment_name}/{ckpt_file.name}"
                        if sync_to_s3(str(ckpt_file), self.s3_bucket, s3_key):
                            print(f"  [S3] Backed up checkpoint_{epoch:03d}.pt")


class GracefulKiller:
    """Handle graceful shutdown on SIGTERM/SIGINT."""

    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        print("\n")
        print("=" * 60)
        print("Received shutdown signal!")
        print("Training will stop after current batch.")
        print("Checkpoints are saved - you can resume later with --resume")
        print("=" * 60)
        self.kill_now = True


def main():
    parser = argparse.ArgumentParser(
        description="AWS Training Wrapper with W&B Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start new training run
    python aws_training/train_wrapper.py --config aws_training/configs/v2_baseline.yaml

    # Resume interrupted training
    python aws_training/train_wrapper.py --config aws_training/configs/v2_baseline.yaml --resume

    # Train without W&B (offline mode)
    python aws_training/train_wrapper.py --config aws_training/configs/v2_baseline.yaml --no-wandb
        """
    )
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--no-s3", action="store_true", help="Disable S3 checkpoint backup")
    parser.add_argument("--s3-bucket", type=str, default="nba-foul-checkpoints-oh",
                        help="S3 bucket for checkpoint backup")
    parser.add_argument("--save-dir", type=str, help="Override save directory")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Generate experiment name and save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_tag = config.get("experiment_tag", "run")
    experiment_name = f"{config['model_arch']}_{experiment_tag}_{timestamp}"

    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = Path("/data/checkpoints") / experiment_name

    # If resuming, find the latest checkpoint directory
    if args.resume and not args.save_dir:
        checkpoint_base = Path("/data/checkpoints")
        if checkpoint_base.exists():
            existing = sorted([d for d in checkpoint_base.iterdir() if d.is_dir()])
            if existing:
                save_dir = existing[-1]
                experiment_name = save_dir.name
                print(f"Resuming from: {save_dir}")

    save_dir.mkdir(parents=True, exist_ok=True)

    # Print configuration
    print("=" * 60)
    print("NBA Foul Detection - AWS Training")
    print("=" * 60)
    print(f"Experiment:    {experiment_name}")
    print(f"Config:        {args.config}")
    print(f"Save dir:      {save_dir}")
    print(f"Model:         {config['model_arch']} + {config['temporal_arch']}")
    print(f"Batch size:    {config['batch_size']}")
    print(f"Epochs:        {config['num_epochs']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"W&B logging:   {'Disabled' if args.no_wandb else 'Enabled'}")
    print(f"S3 backup:     {'Disabled' if args.no_s3 else args.s3_bucket}")
    print("=" * 60)
    print()

    # Setup W&B
    wandb_run = None
    if not args.no_wandb and WANDB_AVAILABLE:
        wandb_run = setup_wandb(config, experiment_name, args.resume)
        if wandb_run:
            print(f"W&B run: {wandb_run.url}")
            print()

    # Setup graceful shutdown handler
    killer = GracefulKiller()

    # Build training command
    cmd = build_train_command(config, str(save_dir), args.resume)

    print("Running command:")
    print(" ".join(cmd))
    print()

    # Setup monitoring
    s3_bucket = None if args.no_s3 else args.s3_bucket
    monitor = TrainingMonitor(save_dir, wandb_run, s3_bucket)
    monitor.start_monitoring()

    # Run training
    start_time = time.time()

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        # Stream output
        for line in iter(process.stdout.readline, ''):
            print(line, end='')

            # Check for graceful shutdown
            if killer.kill_now:
                print("\nSending interrupt to training process...")
                process.send_signal(signal.SIGINT)
                break

        process.wait()
        return_code = process.returncode

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        process.terminate()
        return_code = 1
    finally:
        monitor.stop_monitoring()

    # Final status
    elapsed = time.time() - start_time
    print()
    print("=" * 60)

    if return_code == 0:
        print(f"Training completed successfully!")
        print(f"Total time: {elapsed/3600:.1f} hours")
    else:
        print(f"Training stopped (return code: {return_code})")
        print(f"Elapsed time: {elapsed/3600:.1f} hours")

    print(f"Checkpoints saved to: {save_dir}")
    print("=" * 60)

    # Final checkpoint sync
    monitor.check_progress()

    # Log final model as W&B artifact
    if wandb_run and return_code == 0:
        print("\nUploading model to W&B...")
        artifact = wandb.Artifact(
            f"model-{experiment_name}",
            type="model",
            description=f"Final model from {experiment_name}"
        )
        # Only add the best checkpoint to avoid huge uploads
        best_ckpt = save_dir / "checkpoint_best.pt"
        if best_ckpt.exists():
            artifact.add_file(str(best_ckpt))
        wandb_run.log_artifact(artifact)

    if wandb_run:
        wandb_run.finish()

    print()
    print("Next steps:")
    print("  1. Run evaluation:")
    print(f"     python aws_training/run_evaluation.py --checkpoint {save_dir}")
    print("  2. View results in W&B:")
    print("     https://wandb.ai")
    print("  3. Don't forget to stop your EC2 instance!")


if __name__ == "__main__":
    main()
