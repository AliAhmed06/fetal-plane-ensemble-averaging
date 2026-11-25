# Simple Averaging versus Optimised Fusion in CNN Ensembles for Fetal Ultrasound Plane Classification

Code for the paper of the same title. The study fine-tunes nine ImageNet-pretrained CNNs on the Fetal Planes DB, splits the data by patient, and compares equal-weight averaging with validation-F1 weighting, PSO fusion weights, an equal-budget random search, logistic stacking and a KAN classifier head.

## Data

Download the Fetal Planes DB from Zenodo: https://doi.org/10.5281/zenodo.3904280

The scripts expect the metadata file and images at

```
/kaggle/input/fetal-planes-db/FETAL_PLANES_DB_data.csv
/kaggle/input/fetal-planes-db/Images/
```

Change `RAW_CSV` and `RAW_IMG_DIR` at the top of scripts 01 to 03 if your paths differ.

## Setup

```
pip install -r requirements.txt
```

We ran everything on Kaggle with one GPU.

## Running the pipeline

Run the scripts in order. Outputs go to `/kaggle/working`. Script 01 trains one CNN per run and resumes from its checkpoints, so run it again until all nine are trained. This keeps each run within Kaggle's session limit.

| Script | What it does |
|---|---|
| `01_train_cnns.py` | Builds the patient-level 70/10/20 split (GroupShuffleSplit, seed 42), searches learning rate and weight decay with PSO, fine-tunes the nine CNNs and saves the sample image figure |
| `02_train_kan_head.py` | Extracts features from a frozen ImageNet EfficientNet-B5 and trains the KAN head |
| `03_export_natural_predictions.py` | Rebuilds the split, writes the image list of each split (`natural_preds/train_index.csv`, `val_index.csv`, `test_index.csv`, with image name, label and patient) and saves the softmax outputs of every model on the natural validation and test sets |
| `04_ablation_study.py` | Fusion ablations, KAN versus MLP and linear heads, patient-level bootstrap CIs, paired bootstrap and McNemar tests |
| `05_figures_and_analysis.py` | Confusion matrix, calibration, selective prediction, Grad-CAM, latency and the numbers used in the paper |
| `06_final_figures.py` | Final versions of the class distribution and ablation figures |
| `07_study_design_figure.py` | Study design diagram |

Script 04 is run twice, once with the default top five and once with the five models ranked highest on validation:

```
python 04_ablation_study.py /kaggle/working/audit_config.json
python 04_ablation_study.py configs/reselected_top5.json
```

## Split lists and predictions

The image list of each split, the saved predictions and the frozen EfficientNet-B5 embeddings are in the `natural_preds` folder. Copy it to `/kaggle/working/natural_preds`, or change `PRED_DIR` in the config, and script 04 reproduces the ablation and head comparison tables without retraining. Script 05 also needs the trained checkpoints, and script 06 uses the output of script 05.

## Citation

If you use this code, please cite the paper (details will be added on publication).
