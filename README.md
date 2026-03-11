# 10th ABAW – Valence-Arousal Estimation

## Overview

This repository contains the implementation used for the **Valence-Arousal Estimation Challenge in the 10th Affective Behavior Analysis in-the-Wild (ABAW) Competition**.

The framework focuses on multimodal emotion estimation in unconstrained environments, integrating temporal modeling and cross-modal fusion to predict continuous **valence** and **arousal** signals.

---

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.12+
- CUDA 11+

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

Download the Aff-Wild2 dataset following the official ABAW challenge instructions.

After preprocessing, the dataset directory should be organized as follows:

```pgsql
dataset_root/
 ├── annotations/
 ├── cropped_images/
 ├── cropped_aligned/
 ├── compacted_48/
 ├── raw_videos/
 └── wav/
```
 

Update the dataset path in main.py if necessary.

## Training

With all the default settings in main.py properly configured, run:

```bash
python main.py -folds_to_run 0 -emotion "valence" -stamp "cv"
```

To run multiple folds on multiple machines:

```bash
python main.py -folds_to_run 0 1 2 -emotion "valence" -stamp "cv"
```

If training stops unexpectedly, resume the latest epoch with:

```bash
python main.py -folds_to_run 0 -emotion "valence" -stamp "cv" -resume 1
```

## Output

The results will be saved in the specified -save_path, and will include:

- training logs (.csv)
- trained model checkpoints
- predictions on the unseen partition

Example output structure:

```pgsql
results/
 ├── logs.csv
 ├── checkpoint.pth
 └── predictions/
```

 
## Evaluation

Model performance is evaluated using the Concordance Correlation Coefficient (CCC), which is the official metric used in the ABAW competition.



