# Revision analyses for IMU-D-26-00709

Two pieces of code reproduce every number and figure of the revised manuscript:

* `pancancer_revision.py` writes all tables (repeated cross-validation, sensitivity analyses, negative controls, Cox models, gene re-evaluation).
* `PanCancer_revised_notebook.ipynb` (built by `build_revised_notebook.py`) draws every figure from one stratified 80/20 split. It imports the cohort, labelling and pipeline code from `pancancer_revision.py`, so both use identical definitions and seeds.

Each patient is counted once (one primary-tumor sample, type 01 or 03 for LAML). Every preprocessing and selection step is fitted inside each training fold, on gene expression only. Tumor type and clinical variables are added after PCA.

## Install
    pip install numpy pandas scipy scikit-learn lifelines scikit-survival statsmodels matplotlib nbconvert
    # optional supplementary cells: pip install gseapy shap

Versions used: Python 3.9.6, NumPy 2.0.2, pandas 2.3.3, scikit-learn 1.5.2, lifelines 0.30.0, scikit-survival 0.23.1.

## Data
* Expression: UCSC Xena, TCGA Pan-Cancer (PANCAN), PanCanAtlas EB++ batch-adjusted RSEM matrix (`tcga_gene_expression.tsv`, 20,531 genes x 11,069 samples, log2(x+1)).
* Clinical: Xena `tcga_survival.tsv` (TCGA Pan-Cancer Clinical Data Resource, keyed by sample).

## Run
    python pancancer_revision.py --expr tcga_gene_expression.tsv --clin tcga_survival.tsv --out results --repeats 20
    # quick check: --repeats 1 --sens-repeats 1 --surv-repeats 1 --n-pc 30
    # optional genome-wide adjusted gene screen (slow): --genomewide

    PANCAN_DATA_DIR=/path/to/data PANCAN_CODE_DIR=/path/to/this/folder \
      jupyter nbconvert --to notebook --execute PanCancer_revised_notebook.ipynb

Runtime on a 10-core laptop: about 45 min for the script, 30 min for the notebook.

## Output -> manuscript
| File | Reviewer comment | Manuscript |
|---|---|---|
| R2_4_patient_flow.csv, R1_8_missing_data.csv | R1.8, R2.4 | Table 1, Figure S1, Table S1 |
| R2_5_repeated_cv_summary.csv | R1.3, R2.2, R2.5 | Tables 2, 3, 5, S3 |
| R2_8_paired_differences_cv.csv, R2_8_delong_original_split.csv | R2.8 | Table 4 |
| R1_1_encoding_recovery.csv | R1.1 | Section 7.3, Figure 4C |
| R1_4_per_type_accounting.csv, R1_4_pooled_auc_included_types.csv, R1_4_per_type_auc_cv_summary.csv | R1.4 | Table S4, Figures 4B and 5 |
| R2_1_threshold_sensitivity.csv | R1.2, R2.1 | Table 7 |
| R2_3_ipw_sensitivity.csv, R2_3_retained_vs_excluded.csv, R2_3_exclusion_by_type.csv | R2.3 | Sections 7.5, 8.1, Figure 6 |
| R2_1_9_survival_cv_summary.csv, R2_9_calibration_2y*.csv, R2_9_schoenfeld_tests.csv | R1.6, R2.1, R2.9 | Table 8, Figure S2, Table S5 |
| R2_7_site_within_type.csv, R2_7_variance_partition.csv | R1.5, R2.7 | Section 7.3, Table S6 |
| control_within_type_permutation.csv, control_leave_one_type_out.csv | R2.2 | Section 7.4, Table S7 |
| R2_10_genes_adjusted_stratified_cox.csv; notebook table6_candidate_genes.csv | R1.7, R2.10 | Table 6, Table S8, Figure S3 |
| software_versions.txt | R1.8, minor 7 | Section 4.1 |

`rerun_survival.py` reruns only the Cox block. `pancancer_revision_v1_backup.py` is the pre-revision version, kept for reference.
