# Text-Sentiment-Analysis
# Sentiment Analysis using Logistic Regression

## 📌 Project Overview
This project implements **Sentiment Analysis** on **IMDB movie reviews** using **Natural Language Processing (NLP)** and **Machine Learning**. The model classifies reviews as **positive** or **negative** using **Logistic Regression**.

## 🚀 Features
- **Text Preprocessing**: Tokenization, stopword removal, and lemmatization.
- **Feature Engineering**: TF-IDF vectorization for numerical representation of text.
- **Model Training**: Logistic Regression for sentiment classification.
- **Performance Evaluation**: Precision, Recall, and F1-score.

## 📂 Dataset
- The dataset consists of IMDB movie reviews labeled as **positive** or **negative**.
- The CSV file should have two columns:
  - `review`: The text of the review.
  - `sentiment`: Label ("positive" or "negative").

## 🔧 Installation
To run this project in **Google Colab**, follow these steps:

1. Install required libraries:
```bash
!pip install nltk
```
2. Import necessary dependencies:
```python
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import re
from google.colab import files
```
3. Download necessary NLTK resources:
```python
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```
4. Upload your dataset in Google Colab:
```python
uploaded = files.upload()
file_path = list(uploaded.keys())[0]
df = pd.read_csv(file_path)
```

## 🔍 Preprocessing Steps
- **Remove HTML tags** from reviews.
- **Tokenization**: Split text into individual words.
- **Stopword Removal**: Remove common words like "the", "is", etc.
- **Lemmatization**: Convert words to their base forms.

## 📊 Model Training & Evaluation
- **TF-IDF Vectorization**: Convert text data into numerical format.
- **Train-Test Split**: 80% training, 20% testing.
- **Logistic Regression**: Trained on the dataset.
- **Evaluation Metrics**:
  - **Precision**
  - **Recall**
  - **F1-score**

## 📈 Results
The trained model achieved:
- **Precision**: ~88%
- **Recall**: ~90%
- **F1-score**: ~89%

## 🎯 Future Improvements
- Try other classifiers like **Naïve Bayes** or **Random Forest**.
- Experiment with **word embeddings (Word2Vec, GloVe, BERT)**.
- Tune hyperparameters for better accuracy.

## 🤝 Contributing
Feel free to fork this repository, open issues, or submit pull requests to improve the project.

## 📜 License
This project is open-source and available under the **MIT License**.

