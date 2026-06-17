import json
import shutil
import subprocess
from pathlib import Path

# Paths
metamodel_root = Path.cwd() # Current directory (C:\Project\ode_2023)
project_name = "alzheimer_joint_project"
project_dir = metamodel_root / project_name
runs_dir = project_dir / "runs"

# Clean up old data
if project_dir.exists():
    shutil.rmtree(project_dir)

project_dir.mkdir(parents=True, exist_ok=True)
runs_dir.mkdir(parents=True, exist_ok=True)

# Load training data
print("Loading training data from ad_ode_training_data.json...")
with open("ad_ode_training_data.json", "r") as f:
    training_data = json.load(f)

print(f"Generating {len(training_data)} individual run folders with joint data...")
for i, row in enumerate(training_data):
    run_folder = runs_dir / f"run_{i:04d}"
    run_folder.mkdir(exist_ok=True)
    
    inputs_data = {
        "age_baseline": row["age_baseline"],
        "apoe4_status": row["apoe4_status"],
        "amyloid_baseline": row["amyloid_baseline"],
        "tau_baseline": row["tau_baseline"],
        "amyloid_2yr": row["amyloid_2yr"],
        "tau_2yr": row["tau_2yr"]
    }
    
    outputs_data = {
        "tau_self_dynamic": row["tau_self_dynamic"],
        "amyloid_self_dynamic": row["amyloid_self_dynamic"],
        "amyloid_drive_tau": row["amyloid_drive_tau"],
        "memory_result_baseline": row["memory_result_baseline"],
        "clinical_stage_baseline": row["clinical_stage_baseline"],
        "memory_result_yr5": row["memory_result_yr5"],
        "clinical_stage_yr5": row["clinical_stage_yr5"]
    }
    
    with open(run_folder / "inputs.json", "w") as f:
        json.dump(inputs_data, f)
        
    with open(run_folder / "outputs.json", "w") as f:
        json.dump(outputs_data, f)

print(f"Joint Run Store prepared at: {project_dir}")

import sys

print("\nLaunching Joint Training Process...")
bayesmm_path = Path(sys.executable).parent / "Scripts" / "bayesmm.exe"
if not bayesmm_path.exists():
    bayesmm_path = "bayesmm" # fallback to PATH

cmd = [str(bayesmm_path), "surrogate", "fit", "ad_ode_surrogate.json"]

try:
    process = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print("Success: Surrogate Trained!")
    print(process.stdout)
except subprocess.CalledProcessError as e:
    print("Training failed.")
    print(f"Stdout: {e.stdout}")
    print(f"Stderr: {e.stderr}")