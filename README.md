# Comparative Analysis of CSRNet and MCNN Models for Crowd Counting

## Team Members
- Uroš Dimitrijević
- Miloš Kutlešić

## Project Description
This project addresses the problem of **crowd counting in images**, focusing on a comparative analysis of two neural network architectures:
- **MCNN (Multi-Column Convolutional Neural Network)**
- **CSRNet (Congested Scene Recognition Network)**

The goal is to investigate how these two architectures perform on the task of estimating the number of people in crowd images, using an appropriate crowd counting dataset.

## Motivation
Unlike classical object detection, where each person is localized with a bounding box, the crowd counting approach is better suited for scenes with high people density, partial occlusion, and difficult detection of individual objects.
For this reason, models such as MCNN and CSRNet are more suitable than standard YOLO approaches for this problem.

## Dataset
Dataset used:
- **ShanghaiTech** (Part A & Part B): https://www.kaggle.com/datasets/tthien/shanghaitech

| Part | Images | Description |
|------|--------|-------------|
| Part A | 482 (300 train, 182 test) | Dense crowds, up to ~3000 people per image |
| Part B | 716 (400 train, 316 test) | Sparser scenes, up to ~500 people per image |

Annotations are (x, y) head coordinates stored in `.mat` files.

## Project Goals
The main objectives of the project are:
1. analyze the crowd counting dataset,
2. implement or adapt the **MCNN** and **CSRNet** models,
3. train the models under the same conditions,
4. compare their performance using standard metrics,
5. draw a conclusion about which model gives better results for the problem at hand.

## Research Question
Which model, **MCNN** or **CSRNet**, gives better results on the task of crowd counting in images, in terms of estimation accuracy and stability on the selected dataset?

## Models
### MCNN
MCNN uses multiple parallel convolutional branches with different receptive fields to handle scenes with varying crowd densities.

Paper: [Single-Image Crowd Counting via Multi-Column Convolutional Neural Network](https://openaccess.thecvf.com/content_cvpr_2016/html/Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper.html) (Zhang et al., CVPR 2016)

### CSRNet
CSRNet uses a deeper architecture with dilated convolutions and is known for strong results on crowd counting tasks, particularly in scenes with high crowd density.

Paper: [CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes](https://arxiv.org/abs/1802.10062) (Li et al., CVPR 2018)

## Evaluation Metrics
The following metrics will be used to compare the models:
- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Squared Error)**

Additionally, the following may be considered:
- training time,
- inference time,
- number of model parameters.

## Work Plan
1. Download and analyze the dataset
2. Prepare the data loading and annotation pipeline
3. Implement/adapt the MCNN model
4. Implement/adapt the CSRNet model
5. Train and evaluate the models
6. Comparative analysis of results
7. Write the report and prepare the presentation

## Project Structure
```text
data/       - dataset and data preparation
notebooks/  - exploratory analysis and visualizations
src/        - main project source code
reports/    - notes, images, results, and report drafts
```

## Running the Project
### 1. Clone the repository
```bash
git clone <REPO_LINK>
cd crowd-counting-csrnet-mcnn
```

### 2. Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
```bash
pip install kagglehub
python scripts/download_data.py
```

## Current Status
- [x] topic defined
- [x] dataset selected (ShanghaiTech)
- [x] dataset download script (`scripts/download_data.py`)
- [x] annotation analysis and density-map generation (`src/datasets/density_map.py`, `src/datasets/dataset.py`)
- [x] exploratory notebooks (`notebooks/01_dataset_inspection.ipynb`, `notebooks/02_density_maps.ipynb`)
- [x] dataset loader + normalization/downsample tests
- [ ] MCNN implementation (currently a placeholder in `src/models/mcnn.py`)
- [ ] CSRNet implementation (currently a placeholder in `src/models/csrnet.py`)
- [ ] training and evaluation (`src/train.py`, `src/evaluate.py` are placeholders)
- [ ] final report

## Expected Outcome
By the end of the project, the expected deliverables are:
- a functional implementation of both models,
- an experimental comparison of their performance,
- a clear conclusion about the advantages and disadvantages of the MCNN and CSRNet approaches on the selected dataset.

## Note
This repository is a student project for a machine learning course and serves as an experimental comparative analysis of crowd counting models.