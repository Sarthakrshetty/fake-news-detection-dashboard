
# Fake News Detection Deployment

This project deploys a Machine Learning Fake News Detection model using Streamlit.

## Project Structure

fake_news_detection/
│
├── app.py
├── predict.py
├── train_model_example.py
├── fake_news_model.pkl
├── tfidf_vectorizer.pkl
├── requirements.txt
└── README.md

## Installation

Install dependencies:

pip install -r requirements.txt

## Run the Application

streamlit run app.py

Then open:

http://localhost:8501

## Input

Enter a news article or headline.

## Output

The system will classify the news as:

- Fake News
- Real News

with prediction confidence.

## Notes

Before running the app you must generate:

fake_news_model.pkl  
tfidf_vectorizer.pkl  

You can train them using:

python train_model_example.py
