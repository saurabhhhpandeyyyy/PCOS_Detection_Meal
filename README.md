# PCOS Diagnosis Prediction

## Introduction

Polycystic Ovary Syndrome (PCOS) is a common hormonal disorder that affects women of reproductive age. It is associated with a variety of symptoms, including irregular menstrual cycles, excessive androgen levels, and cysts in the ovaries. Early diagnosis and management of PCOS can improve the overall quality of life and prevent associated complications like infertility, type 2 diabetes, and heart disease.

This project aims to build a machine learning model to predict whether a patient has PCOS based on physical and clinical parameters.

## Project Overview

The project involves the following steps:
1. **Data Collection**: The dataset contains physical and clinical parameters collected from multiple hospitals. It includes both patients with and without PCOS.
2. **Data Preprocessing**: The data undergoes preprocessing steps such as handling missing values, encoding categorical variables, and feature scaling.
3. **Model Building**: Various machine learning algorithms, including XGBoost, are used to build a predictive model.
4. **Evaluation**: The model is evaluated using accuracy, classification reports, confusion matrices, and other metrics.
5. **Single Prediction**: The model can predict the outcome for a single test case, providing both the actual and predicted values along with prediction probabilities.

## Dataset

The dataset consists of 44 physical and clinical parameters such as:
- Age
- BMI (Body Mass Index)
- Blood Pressure (Systolic and Diastolic)
- Hormonal levels (FSH, LH, AMH)
- Menstrual cycle regularity
- Follicle counts
- And other clinical features

### Target Variable:
- **PCOS (Y/N)**: A binary variable indicating whether a patient has been diagnosed with PCOS.

### Features:
A wide range of physical, hormonal, and lifestyle factors are included as features for the predictive model.

## Models Used

- **XGBoost**: The main model used for this project due to its high performance on structured data.
- **Other Models**: Potential models for comparison include Random Forest, Logistic Regression, and Neural Networks.

## Installation

To run this project locally, ensure you have the following dependencies installed:

```bash
pip install numpy pandas scikit-learn xgboost seaborn matplotlib
```

## Running the Project

1. **Data Preprocessing**:
    - Load the dataset and perform necessary preprocessing steps such as filling missing values and scaling the features.

2. **Model Training**:
    - Train the XGBoost model using the preprocessed data.

3. **Evaluation**:
    - Evaluate the model performance using metrics such as accuracy, precision, recall, and F1-score.

4. **Single Prediction**:
    - Use the model to predict the PCOS status for a single test instance, showing both the predicted and actual labels.

## Example

Here is an example of how to train the model and evaluate its performance:

```python
# Load the dataset
data = pd.read_csv('PCOS_data.csv')

# Preprocess the data (handle missing values, scale features, etc.)

# Train the model
xgb_model.fit(X_train, y_train)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
```

## Results

- **Accuracy**: 87.73%
- **Precision**: 89% for non-PCOS, 84% for PCOS
- **Recall**: 93% for non-PCOS, 77% for PCOS

### Confusion Matrix:
The confusion matrix shows the correct and incorrect classifications for both PCOS and non-PCOS cases.

## Future Work

- **Hyperparameter Tuning**: Perform grid search to optimize the model parameters.
- **Feature Engineering**: Explore additional features and interactions that may improve the model's accuracy.
- **Deep Learning**: Investigate the use of neural networks for improved performance.

## Conclusion

This project demonstrates the use of machine learning, particularly XGBoost, in predicting the diagnosis of PCOS based on clinical and physical data. The model shows good predictive performance but can be further improved with feature engineering and hyperparameter tuning.
