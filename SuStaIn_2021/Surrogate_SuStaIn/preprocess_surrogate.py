import json
import shutil
import subprocess
from pathlib import Path

# Paths
metamodel_root = Path(r"C:\Project\Simulating_SuStaIn\Surrogate_SuStaIn")
project_name = "sustain_joint_project"
project_dir = metamodel_root / project_name
runs_dir = project_dir / "runs"

# Clean up old data
if project_dir.exists():
    shutil.rmtree(project_dir)

project_dir.mkdir(parents=True, exist_ok=True)
runs_dir.mkdir(parents=True, exist_ok=True)

# Load training data
print("Loading training data from sustain_training_data.json...")
with open(metamodel_root / "sustain_training_data.json", "r") as f:
    training_data = json.load(f)

# 1. Dynamically write sustain_surrogate.json config
print("Dynamically generating sustain_surrogate.json config based on dataset...")
sample_inputs = list(training_data[0]["inputs"].keys())
sample_outputs = list(training_data[0]["outputs"].keys())

config = {
  "schema_version": "1.0",
  "name": "sustain_surrogate_sbi",
  "kind": "conditional",
  "inputs": sample_inputs,
  "outputs": sample_outputs,
  "backend": "sbi_npe",
  "backend_config": {
    "density_estimator": "maf",
    "max_num_epochs": 80,
    "training_batch_size": 32,
    "learning_rate": 0.0005,
    "validation_fraction": 0.1,
    "stop_after_epochs": 20,
    "show_train_summary": False,
    "summary_samples": 256
  },
  "dataset_ref": {
    "run_store_root": "sustain_joint_project"
  },
  "seed": 123,
  "summary_config": {
    "kind": "index",
    "index": 0
  }
}

config_path = metamodel_root / "sustain_surrogate.json"
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

print(f"Generated config with {len(sample_inputs)} inputs and {len(sample_outputs)} outputs.")

# 2. Build Run Store
print(f"Generating {len(training_data)} individual run folders with joint data...")
for i, row in enumerate(training_data):
    run_folder = runs_dir / f"run_{i:04d}"
    run_folder.mkdir(exist_ok=True)
    
    with open(run_folder / "inputs.json", "w") as f:
        json.dump(row["inputs"], f)
        
    with open(run_folder / "outputs.json", "w") as f:
        json.dump(row["outputs"], f)

print(f"Joint Run Store prepared at: {project_dir}")

# 3. Launch Training
print("\nLaunching Joint Training Process...")
cmd = [r"C:\Users\amitm\anaconda3\envs\bayesian-metamodeling\python.exe", r"C:\Users\amitm\anaconda3\envs\bayesian-metamodeling\Scripts\bayesmm.exe", "surrogate", "fit", "sustain_surrogate.json"]

try:
    process = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=str(metamodel_root))
    print("Success: Surrogate Trained!")
    print(process.stdout)
except subprocess.CalledProcessError as e:
    print("Training failed.")
    print(f"Stdout: {e.stdout}")
    print(f"Stderr: {e.stderr}")
