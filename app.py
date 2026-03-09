
import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.set_page_config(page_title="Fake News Detection", page_icon="📰")

st.title("📰 Fake News Detection System")
st.write("Enter a news article or headline to check whether it is **Fake or Real**.")

news_text = st.text_area("Enter News Text", height=200)

if st.button("Predict"):
    if news_text.strip() == "":
        st.warning("Please enter some news text.")
    else:
        text_vector = vectorizer.transform([news_text])
        prediction = model.predict(text_vector)[0]
        probability = model.predict_proba(text_vector)[0].max()

        if prediction == 1:
            st.error(f"⚠ Fake News Detected (Confidence: {probability:.2f})")
        else:
            st.success(f"✅ Real News (Confidence: {probability:.2f})")
