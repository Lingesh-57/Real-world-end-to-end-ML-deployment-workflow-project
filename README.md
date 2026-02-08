A. Problem Statement:

The objective of this project is to build, evaluate, and deploy multiple machine learning classification models on a real-world dataset. The task involves comparing the performance of traditional machine learning models and ensemble methods using standard evaluation metrics, and deploying the trained models using an interactive Streamlit web application.

B. Dataset Description:

Dataset Name: Breast Cancer Wisconsin (Diagnostic) Dataset
Source: Public dataset (Kaggle - "https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data")
Problem Type: Binary Classification
Number of Instances: 569
Number of Features: 30 numerical features
Target Variable: diagnosis
B → Benign (0)
M → Malignant (1)
The dataset contains computed features from digitized images of breast mass cell nuclei. It is widely used for evaluating classification algorithms in medical diagnosis tasks.

C. Models Used and Evaluation Metrics:

The following six machine learning classification models were implemented and evaluated on the same dataset:

Logistic Regression, Decision Tree Classifier, K-Nearest Neighbors (KNN), Gaussian Naive Bayes, Random Forest (Ensemble), XGBoost (Ensemble)

Evaluation Metrics Used:

Accuracy,AUC Score,Precision,Recall,F1 Score,Matthews Correlation Coefficient (MCC)

C.1 Model Performance Comparison Table:

| ML Model                 | Accuracy   | AUC        | Precision | Recall | F1 Score   | MCC        |
| ------------------------ | ---------- | ---------- | --------- | ------ | ---------- | ---------- |
| Logistic Regression      | 0.9649     | 0.9960     | 0.9750    | 0.9286 | 0.9512     | 0.9245     |
| Decision Tree            | 0.9298     | 0.9246     | 0.9048    | 0.9048 | 0.9048     | 0.8492     |
| KNN                      | 0.9561     | 0.9823     | 0.9744    | 0.9048 | 0.9383     | 0.9058     |
| Naive Bayes              | 0.9386     | 0.9934     | 1.0000    | 0.8333 | 0.9091     | 0.8715     |
| Random Forest (Ensemble) | 0.9737     | 0.9929     | 1.0000    | 0.9286 | 0.9630     | 0.9442     |
| XGBoost (Ensemble)       | 0.9649     | 0.9960     | 1.0000    | 0.9048 | 0.9500     | 0.9258     |

C.2 Observations on Model Performance:
   
| ML Model                 | Observation                                                                                                                  |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression      | Achieved strong performance with high AUC, indicating good class separability for this dataset.                              |
| Decision Tree            | Showed lower performance due to overfitting and high variance when grown without depth constraints.                          |
| KNN                      | Performed well after feature scaling but showed sensitivity to the choice of K and higher computation cost during inference. |
| Naive Bayes              | Achieved perfect precision but lower recall due to the strong independence assumption between features.                      |
| Random Forest (Ensemble) | Achieved the best overall performance with the highest accuracy, F1-score, and MCC due to ensemble averaging.                |
| XGBoost (Ensemble)       | Achieved the highest AUC, demonstrating excellent discrimination capability using boosting and regularization.               |

 
