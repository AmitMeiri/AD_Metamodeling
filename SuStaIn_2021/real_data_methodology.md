# Real Data Simulation Methodology: SuStaIn on ADNI Tau-PET

This document describes the step-by-step processing pipeline used to run the SuStaIn algorithm on real Alzheimer's Disease Neuroimaging Initiative (ADNI) Tau-PET data, matching the methodology established by Vogel et al. (2021).

## 1. Data Sources
We utilized two primary data files:
* **`UCBERKELEY_TAU_6MM_06Sep2026.csv`**: Contains raw, unadjusted Standardized Uptake Value Ratio (SUVR) values and volumetric data for 2,489 Tau-PET scans (across 1,448 subjects). The scans were processed using FreeSurfer v6.0 and the Desikan-Killiany (DK) atlas. The SUVR reference region is the inferior cerebellar gray matter.
* **`ADNI_MASTER_ALL_SUBJECTS.csv`**: Contains corresponding clinical metadata, including the patient diagnosis (`DX`), age, APOE4 status, and ground-truth `VOGEL_SUBTYPE` classifications.

## 2. Defining the 10 Lobar Composites
To avoid overparameterizing the model (using all 68 individual DK regions would yield 204 disease stages), we grouped the cortical and medial temporal regions into **10 bilateral Lobar ROIs** (Frontal, Parietal, Occipital, Lateral Temporal, and Medial Temporal [MTL] for both the left and right hemispheres). 

Because regions vary significantly in size, we calculated the composite lobar SUVR as a **volume-weighted average**:
$$ \text{SUVR}_{\text{Lobe}} = \frac{\sum_{i \in \text{Lobe}} \left( \text{SUVR}_i \times \text{Volume}_i \right)}{\sum_{i \in \text{Lobe}} \text{Volume}_i} $$
where $i$ iterates through all Desikan-Killiany parcels that fall within the specific lobe.

## 3. Z-Score Normalization
SuStaIn operates on normalized Z-scores that reflect the severity of pathology relative to a healthy baseline. 
1. We merged the Tau data with the ADNI clinical master table and filtered for **Cognitively Normal (CN)** individuals (`DX == 'CN'`). This yielded a robust control cohort of 1,378 scans.
2. For each of the 10 lobar regions, we calculated the mean ($\mu_{\text{CN}}$) and standard deviation ($\sigma_{\text{CN}}$) across the healthy control cohort.
3. We then transformed all patient scans into Z-scores:
$$ Z_{\text{subject, ROI}} = \frac{\text{SUVR}_{\text{subject, ROI}} - \mu_{\text{CN, ROI}}}{\sigma_{\text{CN, ROI}}} $$
*Note: Because SuStaIn models a unidirectional cascade of pathology accumulation, any Z-score $< 0$ was clipped to $0$.*

## 4. Running the SuStaIn Algorithm
We executed the `ZscoreSustain` model using the `pySuStaIn` library:
* **Input Features**: The 10 standardized Lobar Z-scores.
* **Severity Thresholds**: We used $Z = \{2, 5, 10\}$ to represent mild, moderate, and severe biomarker events, respectively. This gives 3 events per lobe, totaling a 30-stage temporal progression.
* **MCMC Settings**: The algorithm was allowed to search for up to 4 subtypes (`N_S_max=4`), utilizing 3,000 MCMC iterations with 15 random start points to explore the optimal permutation topology.

## 5. Interpreting the Subtypes
Upon convergence, the SuStaIn algorithm outputs the cross-validation information criterion (CVIC) to identify the optimal number of subtypes. For each subtype, the model generates:
* **The Progression Trajectory**: The sequence array indicating the exact temporal order in which each lobar region reaches $Z=2$, $Z=5$, and $Z=10$. 
* **Population Prevalence**: The fraction of the dataset best described by each subtype.
* **Individual Classifications**: Joint posterior probabilities assigning each subject to their most likely subtype and current disease stage.
