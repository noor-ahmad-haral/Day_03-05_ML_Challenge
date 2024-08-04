# Day 3: Support Vector Machines (SVM)

Welcome to Day 3 of the 5-Day Machine Learning Revision Challenge! Today, we'll explore Support Vector Machines (SVM), a powerful algorithm for classification tasks.

## Objective

Understand and implement SVM for classifying data, specifically for spam email detection.

## Algorithm Overview

Support Vector Machines (SVM) are supervised learning models used for classification and regression tasks. SVM aims to find the optimal hyperplane that separates classes by maximizing the margin between them.

### How It Works

- **Optimal Hyperplane:** Identifies the best boundary to divide the data into classes.
- **Margin:** The distance between the hyperplane and the nearest data points from each class. SVM maximizes this margin for better separation.
- **Support Vectors:** The data points closest to the hyperplane, crucial in defining its position.

### Why Use It?

- **High-Dimensional Spaces:** Effective in spaces with many features.
- **Clear Margin of Separation:** Performs well with clear boundaries between classes.
- **Versatility:** Handles both linear and non-linear classification using kernel functions.

## Project: Spam Email Classification

### Tasks

1. **Load and Preprocess the Dataset:**
   - Import necessary libraries.
   - Load the dataset and clean the data.
   - Split the dataset into training and testing sets.

2. **Convert Text to Numerical Features Using TF-IDF:**
   - Transform text data into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency).

3. **Implement and Train an SVM Classifier:**
   - Train an SVM model on the training data.

4. **Evaluate Performance:**
   - Assess the model using precision, recall, and F1-score.
