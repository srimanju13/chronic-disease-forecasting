# Prediction of Chronic Diseases: Methods, Findings, and Recommendations

## Description
This project aims to predict the likelihood of individuals developing chronic diseases using machine learning models. The analysis is based on the U.S. Chronic Disease Indicators dataset, focusing on early detection and prevention to reduce healthcare costs and improve patient outcomes.

## Project Overview

This project demonstrates a complete machine learning workflow using Python and scikit-learn. The project is modularized for readability, reuse, and easier collaboration. It includes steps for data loading, preprocessing, model training and evaluation, and utilities for EDA and visualization.

## Folder Structure

```
final-project/
├── README.md                                        # Project overview and instructions
├── requirements.txt                                 # List of required Python packages
├── main.py                                          # Main entry point to run the workflow
├── notebooks/          
│   └── final_alg.ipynb                              # Jupyter notebook for model analysis
└── data/
    └── U.S_Chronic_Disease_Indicators_CDI_2023.csv  # Dataset
└── src/
    ├── data_loader.py                               # Data loading module
    ├── preprocessing.py                             # Data preprocessing module
    ├── model.py                                     # Model creation and training module
    └── utils.py                                     # Utility functions
└── assets/
    └── Findings_and_Recommendations.pptx            # Presentation with findings and recommendations


## How to Run

1. Clone this repository.
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the workflow:
   ```bash
   python main.py
   ```
4. Alternatively, explore the `notebooks/final_alg.ipynb` notebook.

## Features

- Clean modular code
- Data loading with error handling
- Preprocessing: missing value handling, label encoding, scaling
- Model training with RandomForestClassifier
- Model evaluation with accuracy score and classification report
- Feature importance visualization

## Features

Clean, modular code for ease of maintenance and reuse.
Data loading with error handling for robust performance.
Preprocessing steps including missing value handling, label encoding, and feature scaling.
Model training using the RandomForestClassifier.
Model evaluation with metrics such as accuracy score and classification report.
Feature importance visualization to interpret model predictions.

## Requirements

See `requirements.txt` for a list of Python packages required to run this project.

## Author

Naga Sri Manju Gajula
