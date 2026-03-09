import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load model
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Label mapping
label_map = {
    "bs": "Fake News",
    "bias": "Biased News",
    "conspiracy": "Conspiracy Theory",
    "hate": "Hate Speech",
    "satire": "Satirical News",
    "state": "State Propaganda",
    "junksci": "Junk Science",
    "fake": "Fake News",
    "reliable": "Reliable News"
}

# Page config
st.set_page_config(
    page_title="Fake News AI Dashboard",
    page_icon="🧠",
    layout="wide"
)

# ---------- THEME ----------
st.markdown("""
<style>

.stApp {
background: linear-gradient(135deg,#0f172a,#1e293b);
color:white;
}

section[data-testid="stSidebar"] {
background:#020617;
}

section[data-testid="stSidebar"] * {
color:white !important;
}

/* FIX METRICS */

[data-testid="stMetricValue"] {
color:#ffffff !important;
font-size:34px;
font-weight:bold;
}

[data-testid="stMetricLabel"] {
color:#cbd5f1 !important;
}

/* HERO BANNER */

.hero-banner {
background: linear-gradient(90deg,#020617,#1e3a8a);
padding:30px;
border-radius:12px;
text-align:center;
margin-bottom:20px;
}

/* NEWS CARDS */

.news-card {
background:#1e293b;
padding:18px;
border-radius:10px;
text-align:center;
font-size:18px;
}

</style>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
st.sidebar.title("🧠 Fake News AI")

menu = st.sidebar.radio(
    "Navigation",
    ["Home","Detect News","Analytics","Model Info"]
)

# ---------- HOME ----------
if menu == "Home":

    st.markdown("""
    <div class="hero-banner">
    <h1 style="color:white;">📰 Fake News Detection AI Platform</h1>
    <p style="color:#cbd5f1;font-size:18px;">
    AI-powered misinformation analysis using Machine Learning & NLP
    </p>
    </div>
    """, unsafe_allow_html=True)

    # Moving Breaking News
    st.markdown("""
    <div style="background:#ef4444;
    color:white;
    padding:10px;
    border-radius:6px;
    font-weight:bold;
    margin-bottom:20px;">

    <marquee behavior="scroll" direction="left" scrollamount="6">

    🚨 Breaking: AI detecting misinformation patterns • Election propaganda analysis • Vaccine rumor detection • Media bias tracking • Global conspiracy monitoring

    </marquee>

    </div>
    """, unsafe_allow_html=True)

    col1,col2,col3 = st.columns(3)

    col1.metric("Algorithms Used","8")
    col2.metric("Best Model","Random Forest")
    col3.metric("Accuracy","~91%")

    st.write("---")

    st.subheader("AI News Intelligence")

    st.write("""
This AI system analyzes news articles and detects misinformation patterns
using **Machine Learning and Natural Language Processing (NLP)**.

Workflow:

Dataset → Preprocessing → TF-IDF → ML Model → Prediction
""")

    st.write("")

    st.subheader("Trending Fake News Topics")

    col1,col2,col3 = st.columns(3)

    col1.markdown('<div class="news-card">📰 Political Propaganda</div>', unsafe_allow_html=True)
    col2.markdown('<div class="news-card">🧬 Health & Vaccine Rumors</div>', unsafe_allow_html=True)
    col3.markdown('<div class="news-card">🌍 Global Conspiracy</div>', unsafe_allow_html=True)

# ---------- DETECT NEWS ----------
elif menu == "Detect News":

    st.title("🔎 Analyze News Article")

    news_text = st.text_area("Paste news article here", height=200)

    if st.button("Analyze News"):

        if news_text.strip()=="":
            st.warning("Please enter text")

        else:

            vector = vectorizer.transform([news_text])

            prediction = model.predict(vector)[0]

            probabilities = model.predict_proba(vector)[0]

            confidence = np.max(probabilities)

            prediction_text = label_map.get(prediction,prediction)

            st.success(f"Prediction: {prediction_text}")

            st.info(f"Confidence Score: {confidence:.2f}")

            classes = model.classes_

            df = pd.DataFrame({
                "Category":[label_map.get(c,c) for c in classes],
                "Probability":probabilities
            })

            fig = px.bar(
                df,
                x="Category",
                y="Probability",
                color="Probability",
                title="Prediction Probability Distribution"
            )

            st.plotly_chart(fig,use_container_width=True)

# ---------- ANALYTICS ----------
elif menu == "Analytics":

    st.title("📊 Dataset & Model Analytics")

    st.subheader("Machine Learning Model Comparison")

    model_accuracy = {
        "Naive Bayes":0.89,
        "Logistic Regression":0.898,
        "SVM":0.908,
        "Decision Tree":0.87,
        "Random Forest":0.915,
        "Gradient Boosting":0.90,
        "KNN":0.86,
        "ANN":0.91
    }

    df_models = pd.DataFrame({
        "Model":list(model_accuracy.keys()),
        "Accuracy":list(model_accuracy.values())
    })

    fig1 = px.bar(
        df_models,
        x="Model",
        y="Accuracy",
        color="Accuracy",
        title="Model Accuracy Comparison"
    )

    st.plotly_chart(fig1,use_container_width=True)

    st.subheader("Fake vs Real Distribution")

    categories = ["Fake","Real"]
    counts = [12000,3500]

    df_fake_real = pd.DataFrame({
        "Category":categories,
        "Count":counts
    })

    fig2 = px.bar(
        df_fake_real,
        x="Category",
        y="Count",
        color="Category"
    )

    st.plotly_chart(fig2,use_container_width=True)

    st.subheader("Trending Fake News Keywords")

    text_data = """
    fake conspiracy propaganda hoax rumor election vaccine
    misinformation media scandal politics secret government
    """

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="black"
    ).generate(text_data)

    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")

    st.pyplot(fig)

# ---------- MODEL INFO ----------
elif menu == "Model Info":

    st.title("🤖 Model Information")

    st.write("""
Machine Learning Models Implemented:

• Naïve Bayes  
• Logistic Regression  
• Support Vector Machine  
• Random Forest  
• Decision Tree  
• Gradient Boosting  
• K-Nearest Neighbors  
• Artificial Neural Network
""")

    st.success("Best Performing Model: Random Forest")
