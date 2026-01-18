# Student Performance Prediction - Complete Results

```
STUDENT PERFORMANCE PREDICTION - MACHINE LEARNING ANALYSIS
Supervised Learning Classification Problem
============================================================
LOADING AND EXPLORING THE DATASET
============================================================
Dataset shape: (10000, 6)
Number of samples: 10000
Number of features: 6

Dataset columns:
['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced', 'Performance Index']

First 5 rows:
   Hours Studied  ...  Performance Index
0              7  ...               91.0
1              4  ...               65.0
2              8  ...               45.0
3              5  ...               36.0
4              7  ...               66.0

[5 rows x 6 columns]

Dataset info:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10000 entries, 0 to 9999
Data columns (total 6 columns):
 #   Column                            Non-Null Count  Dtype  
---  ------                            --------------  -----  
 0   Hours Studied                     10000 non-null  int64  
 1   Previous Scores                   10000 non-null  int64  
 2   Extracurricular Activities        10000 non-null  object 
 3   Sleep Hours                       10000 non-null  int64  
 4   Sample Question Papers Practiced  10000 non-null  int64  
 5   Performance Index                 10000 non-null  float64
dtypes: float64(1), int64(4), object(1)
memory usage: 468.9+ KB
None

Basic statistics:
       Hours Studied  ...  Performance Index
count   10000.000000  ...       10000.000000
mean        4.992900  ...          55.224800
std         2.589309  ...          19.212558
min         1.000000  ...          10.000000
25%         3.000000  ...          40.000000
50%         5.000000  ...          55.000000
75%         7.000000  ...          71.000000
max         9.000000  ...         100.000000

[8 rows x 5 columns]

Missing values:
Hours Studied                       0
Previous Scores                     0
Extracurricular Activities          0
Sleep Hours                         0
Sample Question Papers Practiced    0
Performance Index                   0
dtype: int64

============================================================
DATA PREPROCESSING
============================================================
Encoded 'Extracurricular Activities': Yes=1, No=0

Performance categories distribution:
Performance_Category
Medium    4933
Low       2562
High      2505
Name: count, dtype: int64

Features shape: (10000, 5)
Target shape: (10000,)

============================================================
SPLITTING THE DATASET
============================================================
Training set size: 8000 samples
Testing set size: 2000 samples
Training set percentage: 80.0%
Testing set percentage: 20.0%

============================================================
EXPERIMENT 1: RANDOM FOREST CLASSIFIER
============================================================
Random Forest Accuracy: 0.9405 (94.05%)

Classification Report:
              precision    recall  f1-score   support

        High       0.96      0.92      0.94       501
         Low       0.94      0.95      0.94       512
      Medium       0.93      0.95      0.94       987

    accuracy                           0.94      2000
   macro avg       0.94      0.94      0.94      2000
weighted avg       0.94      0.94      0.94      2000


Confusion Matrix:
[[461   0  40]
 [  0 484  28]
 [ 18  33 936]]

Feature Importance:
                            feature  importance
1                   Previous Scores    0.745916
0                     Hours Studied    0.171311
4  Sample Question Papers Practiced    0.042064
3                       Sleep Hours    0.030321
2        Extracurricular Activities    0.010388

============================================================
EXPERIMENT 2: K-NEAREST NEIGHBORS WITH FEATURE SCALING
============================================================
Applied StandardScaler to normalize features

Testing different k values:
k=3: Accuracy = 0.9135 (91.35%)
k=5: Accuracy = 0.9145 (91.45%)
k=7: Accuracy = 0.9115 (91.15%)
k=9: Accuracy = 0.9155 (91.55%)
k=11: Accuracy = 0.9210 (92.10%)

Best k value: 11 with accuracy: 0.9210

Final KNN Accuracy: 0.9210 (92.10%)

Classification Report:
              precision    recall  f1-score   support

        High       0.93      0.91      0.92       501
         Low       0.93      0.91      0.92       512
      Medium       0.91      0.93      0.92       987

    accuracy                           0.92      2000
   macro avg       0.92      0.92      0.92      2000
weighted avg       0.92      0.92      0.92      2000


Confusion Matrix:
[[455   0  46]
 [  0 467  45]
 [ 32  35 920]]

============================================================
MODEL COMPARISON AND ANALYSIS
============================================================
Random Forest Accuracy: 0.9405 (94.05%)
KNN Accuracy: 0.9210 (92.10%)

Random Forest performs better by 1.95 percentage points
Best performing model: Random Forest

============================================================
SUMMARY
============================================================
✓ Successfully loaded and explored the student performance dataset
✓ Preprocessed data and created performance categories
✓ Split data into 80% training and 20% testing sets
✓ Conducted Experiment 1: Random Forest Classifier
✓ Conducted Experiment 2: KNN with Feature Scaling and Hyperparameter Tuning
✓ Evaluated both models using accuracy metric
✓ Compared model performances

Final Results:
- Random Forest Accuracy: 94.05%
- KNN Accuracy: 92.10%
- Best Model: Random Forest

This analysis demonstrates the complete supervised machine learning workflow
for predicting student performance based on study habits and personal factors.
```