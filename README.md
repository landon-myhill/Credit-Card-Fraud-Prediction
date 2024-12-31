# Credit Card Fraud Detection

This repository contains an R implementation for predicting credit card fraud using logistic regression. The dataset is preprocessed, and multiple statistical and visual analyses are performed to enhance predictive performance. The implementation includes metrics evaluation, residual diagnostics, and visualization of model assumptions.

## Project Overview

The goal of this project is to classify transactions as fraudulent or non-fraudulent based on features such as transaction amount, customer age, gender, transaction month, and category. Logistic regression is used as the primary modeling approach, and key metrics like accuracy, sensitivity, specificity, and AUC are calculated to assess the model's performance.

## Features

- **Data Preprocessing**:
  - Conversion of date of birth into age groups.
  - Extraction of transaction month from timestamps.
  - Handling of missing values.
  - Log transformation for skewed transaction amounts.

- **Modeling**:
  - Logistic regression to predict fraudulent transactions.
  - Residual diagnostics for model assumptions.
  - Multicollinearity checks with Variance Inflation Factor (VIF).

- **Performance Evaluation**:
  - Confusion matrix generation.
  - Metrics such as accuracy, sensitivity (recall), specificity, and F1-score.
  - ROC curve plotting and calculation of AUC.

- **Visualization**:
  - Distribution of transaction amounts (log-transformed and raw).
  - Boxplots of transaction amounts by category.
  - Diagnostic plots for residuals and Cook's distance.

## Installation

To run this project, ensure you have R installed along with the required libraries:

```R
install.packages(c("lubridate", "dplyr", "pROC", "tidyr", "car", "ggplot2", "ResourceSelection"))
