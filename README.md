 #  IMDb Movie Reviews - Sentiment Analysis Project

This project uses **Natural Language Processing (NLP)** and **supervised machine learning** to classify IMDb movie reviews as either **positive** or **negative**. 

---

##  Project Overview

- **Objective**: Build a binary classifier that predicts the sentiment of a movie review.
- **Dataset**: [IMDb Large Movie Review Dataset (Kaggle)](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)  
- **Best Model**: Logistic Regression + TF-IDF Vectorization
- **Achieved Accuracy**: ~90.29%

---

##  Workflow

### 1. Exploratory Data Analysis (EDA)
- Analyzed distribution of review lengths
- Generated **WordClouds** for both positive and negative reviews
- Verified **class balance** (almost equal distribution)

### 2. Preprocessing
- Converted text to lowercase
- Removed punctuation and non-alphabetic characters
- Removed stopwords
- Applied tokenization
- Applied **lemmatization**
- Used **TF-IDF vectorization** to convert text to numerical features

### 3. Modeling
Built and evaluated the following models:
- Logistic Regression *(best performing model)*
- Naive Bayes
- Random Forest Classifier

### 4. Evaluation Metrics

| Metric         | Score     |
|----------------|-----------|
| Accuracy       | 90.29%    |
| Precision      | 90%       |
| Recall         | 91%       |
| F1-Score       | 90%       |

Additional:
-  Confusion Matrix (visualized)
-  Classification Report

---

##  Tech Stack

- **Python**
- **Pandas**, **NumPy**
- **Scikit-learn**
- **NLTK**, **SpaCy**
- **Matplotlib**, **Seaborn**
- **WordCloud**

---

##  Dataset

You can find the dataset used here:  
🔗 [IMDb Movie Review Dataset on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

---

