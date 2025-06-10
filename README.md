# Customer Churn Prediction Project

This project focuses on predicting customer churn using various machine learning models. Customer churn prediction is crucial for businesses to identify and retain customers who might discontinue their services.

## Project Overview

The project implements multiple machine learning models to predict customer churn, including:
- Random Forest
- Logistic Regression
- Decision Tree
- Ada Boosting
- Gradient Boosting

Each model's performance is evaluated using confusion matrices and other metrics to determine the most effective approach for churn prediction.

## Repository Structure

```
├── data/
│   ├── customer_churn_dataset-testing-master.csv   # Testing dataset
│   ├── customer_churn_dataset-training-master.csv  # Training dataset
│   └── data_info.md                               # Dataset information
├── membangun_model/
│   ├── modelling_tuning.py                        # Model tuning implementation
│   ├── modelling.py                               # Main modeling implementation
│   ├── requirements.txt                           # Project dependencies
│   └── images/                                    # Model evaluation visualizations
│       ├── confusion_matrix_*.png                 # Confusion matrices for each model
├── models/
│   └── preprocessor.pkl                           # Saved preprocessor model
└── preprocessing_dataset/                          # Preprocessed data
    ├── test_preprocessing.csv
    └── train_preprocessing.csv
```

## Dataset

The dataset contains customer information and their churn status. It includes various features that might influence customer churn, such as customer demographics, usage patterns, and service-related information.

Source: [Kaggle - Customer Churn Dataset](https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset)

## Dependencies

The project requires the following main dependencies:
- Python 3.x
- MLflow 2.22.0
- NumPy 1.26.4
- Pandas 2.2.3
- Scikit-learn 1.6.1
- Matplotlib 3.10.3
- Seaborn 0.13.2

To install all dependencies, run:
```bash
pip install -r membangun_model/requirements.txt
```

## Model Implementation

The project implements several machine learning models to compare their performance in predicting customer churn:

1. **Random Forest**: Ensemble learning method for classification
2. **Logistic Regression**: Both basic and optimized versions
3. **Decision Tree**: Tree-based classification model
4. **Ada Boosting**: Adaptive boosting ensemble method
5. **Gradient Boosting**: Gradient boosting machines implementation

Model evaluation results are visualized through confusion matrices, which can be found in the `membangun_model/images/` directory.

## Model Preprocessing

Data preprocessing steps are implemented and saved using scikit-learn's preprocessing tools. The preprocessor is saved as `preprocessor.pkl` in the models directory for consistent preprocessing across training and inference.

## Getting Started

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r membangun_model/requirements.txt
   ```
3. Run the modeling scripts:
   ```bash
   python membangun_model/modelling.py
   python membangun_model/modelling_tuning.py
   ```

## Results

The model evaluation results, including confusion matrices and performance metrics, are stored as images in the `membangun_model/images/` directory. These visualizations help in comparing the performance of different models and selecting the most effective approach for churn prediction.