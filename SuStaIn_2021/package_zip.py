import os
import shutil
import zipfile
from pathlib import Path

base_dir = Path("C:/Project/AD_Metamodeling/SuStaIn_2021")
staging_dir = base_dir / "sustain_real_data_package"
zip_output_path = base_dir / "sustain_real_data_package.zip"

if staging_dir.exists():
    shutil.rmtree(staging_dir)
staging_dir.mkdir(parents=True, exist_ok=True)

# 1. Write the README.md for the package
readme_content = """# SuStaIn Real Data Analysis Package (ADNI Tau-PET)

This package contains all datasets, scripts, trained models, visual diagrams, and patient classifications for the unsupervised SuStaIn disease progression analysis on real ADNI Tau-PET scans.

---

## Package Contents

### 1. Data Files
* `ADNI_MASTER_ALL_SUBJECTS.csv`: Master clinical metadata, including diagnoses (`DX`), visit codes (`VISCODE2`), and demographics.
* `UCBERKELEY_TAU_6MM_06Sep2026.csv`: Desikan-Killiany (DK) ROI measurements (`_SUVR` and `_VOLUME`) referenced to inferior cerebellar gray matter.

### 2. Execution Scripts
* `run_sustain_real_data.py`: The complete end-to-end data pipeline. Groups 68 DK regions into 10 volume-weighted lobar ROIs, computes Z-scores against Healthy Controls (`DX == 'CN'`), runs 5-Fold Cross-Validation (CVIC) up to 6 subtypes, and trains the final model.
* `generate_clean_csvs.py`: Formats and exports the 1-based patient classifications for both 4- and 6-subtype models.
* `analyze_final_models.py`: Inspects the fitted MCMC sequence matrices and computes cross-tabulations.

### 3. Patient Classification Spreadsheets
* `real_data_classification_4subtypes.csv`: Per-scan classifications for the canonical **4-subtype model** (matching Vogel et al., 2021), with explicit labels for Stage 0 (Tau-Negative) and Subtypes 1-4.
* `real_data_classification_6subtypes.csv`: Per-scan classifications for the **6-subtype model** (data-driven global CVIC minimum), with explicit labels for Stage 0 (Tau-Negative) and Subtypes 1-6.

### 4. Detailed Documentation
* `real_data_methodology.md`: Mathematical methodology covering volume-weighted lobar aggregation, healthy control Z-scoring, and SuStaIn MCMC parameters.
* `real_data_subtypes_analysis.md`: In-depth analysis of the discovered biological trajectories, CVIC cross-validation scores, and comparison to Vogel's 4 published subtypes.

### 5. Output Artifacts Folder (`real_data_output/`)
* `pickle_files/`: All serialized Python pickle model artifacts across all 5 cross-validation folds and final full-dataset models (`subtype0` through `subtype5`).
* `adni_tau_real_subtype*_PVD.png_all-subtypes.png`: Positional Variance Diagrams (PVDs) visualizing the temporal biomarker cascades across subtypes.
* `real_data_classification.csv`: Raw output classification matrix.

---

## How to Run / Replicate
To rerun the pipeline in Python:
```bash
python run_sustain_real_data.py
```
"""

(staging_dir / "README.md").write_text(readme_content, encoding="utf-8")

# 2. Copy files into staging
files_to_copy = [
    "ADNI_MASTER_ALL_SUBJECTS.csv",
    "UCBERKELEY_TAU_6MM_06Sep2026.csv",
    "run_sustain_real_data.py",
    "generate_clean_csvs.py",
    "analyze_final_models.py",
    "real_data_methodology.md",
    "real_data_subtypes_analysis.md",
    "real_data_output/real_data_classification_4subtypes.csv",
    "real_data_output/real_data_classification_6subtypes.csv",
]

for f in files_to_copy:
    src = base_dir / f
    dst = staging_dir / Path(f).name
    if src.exists():
        shutil.copy2(src, dst)
        print(f"Copied {f}")
    else:
        print(f"Warning: {f} not found!")

# 3. Copy entire real_data_output folder
src_output_dir = base_dir / "real_data_output"
dst_output_dir = staging_dir / "real_data_output"
if src_output_dir.exists():
    shutil.copytree(src_output_dir, dst_output_dir)
    print("Copied real_data_output directory")

# 4. Create zip archive
print(f"\nCompressing into {zip_output_path}...")
with zipfile.ZipFile(zip_output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(staging_dir):
        for file in files:
            file_path = Path(root) / file
            archive_name = file_path.relative_to(staging_dir)
            zipf.write(file_path, archive_name)

# Cleanup staging directory
shutil.rmtree(staging_dir)

zip_size_mb = zip_output_path.stat().st_size / (1024 * 1024)
print(f"\nSuccess! Created {zip_output_path} ({zip_size_mb:.2f} MB)")
