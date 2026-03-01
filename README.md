# Mask R-CNN Branch  
Improving Combined Detection and Classification of TEM Defects

This branch contains the implementation of the Mask R-CNN pipeline used for defect detection and classification, along with utilities for mask simulation, annotation processing, and performance evaluation.

---

## 📁 Branch Overview

This branch includes:

### Core Training & Inference
- `detectron_maskrcnn.py`  
  Mask R-CNN model configuration and training setup using Detectron2.

- `run_detectron.py`  
  Script for running training and/or inference on the prepared dataset.

---

### Annotation & Mask Processing
- `load_annotations.py`  
  Loads dataset annotations and prepares them in Detectron2-compatible format.

- `mask2anno_filter.py`  
  Converts segmentation masks into annotation format and optionally applies filtering.

- `circle_oval_dpi.ipynb`  
  Notebook for analyzing defect morphology statistics (ellipse fitting, size/orientation distributions) and simulating masks.

---

### Evaluation Scripts
- `analyze_maskrcnn_class.py`  
  Computes detection and classification performance metrics (including F1 scores).

- `analyze_maskrcnn_nofilter.py`  
  Evaluation without post-processing filters.


---

## 🧠 Purpose of This Branch

This branch is responsible for:

- Simulating defect masks based on statistical distributions
- Training Mask R-CNN models on:
  - Experimental data only (EXP)
  - Experimental + Generated data (EXP+GEN)
- Evaluating performance using:
  - `F1_detect`
  - `F1_class`

---

## 📂 Dataset and Data Splits

Two independent dataset splits were used:

---

## ⚙️ Installation

### Requirements

- Python 
- PyTorch
- Detectron2
- NumPy
- OpenCV
- Matplotlib (for analysis scripts)
