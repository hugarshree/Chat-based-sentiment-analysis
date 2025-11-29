# Chat-based-sentiment-analysis
Chat-Based Sentiment Analysis using XGBoost

A machine-learning project that predicts the sentiment (Positive / Negative / Neutral) of chat messages using XGBoost, text preprocessing, and vectorization.

🚀 Project Overview

This project processes chat messages, converts the text into numerical features using TF-IDF or Bag-of-Words, and uses an XGBoost classifier to classify sentiment.
It can be integrated into chat apps, customer support tools, or feedback systems.

🛠️ Features

✔️ Clean data preprocessing
✔️ Text normalization (lowercase, tokenization, stopword removal)
✔️ TF-IDF vectorizer
✔️ XGBoost model training
✔️ Model evaluation (accuracy, precision, recall, F1-score)
✔️ Prediction on new chat messages

📦 Requirements

Add this to requirements.txt:

xgboost
scikit-learn
pandas
numpy
nltk



Run:
pip install -r requirements.txt

📝 Dataset

Your dataset should contain:

message	sentiment
"I love this app!"	positive
"Worst service"	negative
"okay fine"	neutral

🔧 How the Model Works
1️⃣ Preprocessing

Convert to lowercase
Remove punctuation
Remove stopwords
Lemmatization (optional)
Convert to vector (TF-IDF)

2️⃣ Training with XGBoost
from xgboost import XGBClassifier

model = XGBClassifier(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=200,
    objective='multi:softmax',
    num_class=3
)
model.fit(X_train, y_train)

3️⃣ Prediction
pred = model.predict(vectorizer.transform(["This is awesome"]))

📊 Evaluation

Example metrics printed after training:

Accuracy  : 89%
Precision : 87%
Recall    : 88%
F1-score  : 87%

▶️ How to Run
1. Train the model
python src/train_model.py

2. Predict sentiment from user chat
python src/predict.py "Service was very slow"


Output:
Sentiment: negative

📌 Future Improvements

Add deep learning (Bi-LSTM / BERT)
Deploy as API using FastAPI
Add emoji & multilingual support
Build a simple chat UI
