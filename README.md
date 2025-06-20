#  Movie Review Sentiment Analysis

##  Project Overview

This project focuses on analyzing the **sentiment** of movie reviews using machine learning techniques. By classifying user reviews as **positive** or **negative**, we can gain insights into audience reception and build tools for feedback analysis in the entertainment industry.

The project includes preprocessing raw text, feature extraction using TF-IDF, and training multiple classifiers to determine the best-performing sentiment analyzer.

---

##  Objectives

- Clean and preprocess movie reviews
- Convert raw text into numerical vectors using TF-IDF
- Train and compare ML models (Naive Bayes, Logistic Regression, SVM)
- Visualize results and metrics
- Deploy as a user-friendly app (optional)

---

## 📂 Dataset

Source: [IMDb Dataset of 50K Movie Reviews on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)  
Type: Binary Sentiment Classification  
Size: 50,000 labeled reviews  
- 25,000 for training (balanced)  
- 25,000 for testing (balanced)  

**Labels:**  
- `1` – Positive  
- `0` – Negative 

---

##  Technologies Used

- **Python**
  - `pandas`, `numpy` – data handling
  - `nltk`, `re`, `string` – text preprocessing
  - `scikit-learn` – machine learning models and metrics
  - `matplotlib`, `seaborn`, `wordcloud` – visualization

---

##  Project Workflow

1. **Data Loading**
   - Import and merge datasets
   - Assign sentiment labels

2. **Text Preprocessing**
   - Lowercasing, punctuation removal, stopword filtering
   - Lemmatization

3. **Feature Extraction**
   - TF-IDF Vectorizer for text representation

4. **Model Training**
   - Train Naive Bayes, Logistic Regression, and Support Vector Machine
   - Evaluate using Accuracy, F1-Score, Confusion Matrix

5. **Visualization**
   - Word Clouds for positive/negative reviews
   - Confusion matrices and classification reports

---

##  Results

| Model                 | Accuracy |
|----------------------|----------|
| Logistic Regression  | 86.8%    |
| Naive Bayes          | 84.2%    |
| Support Vector Classifier | 85.1%    |

- **Best Performing Model**: Logistic Regression
- Word clouds showed that negative reviews often included words like *boring, worst*, while positive ones had *amazing, brilliant, loved*.

---


