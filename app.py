import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px

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

# Page configuration
st.set_page_config(
    page_title="Fake News Detection Dashboard",
    page_icon="📰",
    layout="wide"
)

# ------------------- THEME -------------------

st.markdown("""
<style>

/* Main dashboard background */
.stApp {
background: linear-gradient(135deg,#1e293b,#0f172a);
color:#ffffff;
}

/* Sidebar color */
section[data-testid="stSidebar"] {
background: #0f172a;
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
color:#ffffff !important;
}

/* Titles */
h1, h2, h3 {
color:#f1f5f9;
}

/* Paragraph text */
p, div {
color:#e2e8f0;
}

/* Metric cards */
.metric-card {
background: linear-gradient(135deg,#3b82f6,#1d4ed8);
padding:20px;
border-radius:12px;
text-align:center;
font-weight:bold;
color:white;
}

/* Buttons */
.stButton>button {
background-color:#2563eb;
color:white;
border-radius:8px;
border:none;
padding:8px 16px;
}

.stButton>button:hover {
background-color:#1d4ed8;
color:white;
}

</style>
""", unsafe_allow_html=True)

# ------------------- SIDEBAR -------------------

st.sidebar.title("🧠 Fake News AI Dashboard")

menu = st.sidebar.radio(
    "Navigation",
    ["Home","Detect News","Analytics","Model Info"]
)

# ------------------- HOME -------------------

if menu == "Home":

    st.title("📰 Fake News Detection System")

    col1,col2,col3 = st.columns(3)

    col1.metric("Algorithms Used","8")
    col2.metric("Best Model","Random Forest")
    col3.metric("Accuracy","~91%")

    st.write("---")

    st.write("""
This system analyzes news articles and detects misinformation patterns using **Machine Learning and NLP**.

Workflow:

Dataset → Preprocessing → TF-IDF → ML Model → Prediction
""")

# ------------------- DETECT NEWS -------------------

elif menu == "Detect News":

    st.title("🔎 Analyze News Article")

    news_text = st.text_area("Paste News Article Here", height=200)

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
                title="Prediction Probability Distribution",
                color_continuous_scale="Blues"
            )

            st.plotly_chart(fig,use_container_width=True)

# ------------------- ANALYTICS -------------------

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
        title="Model Accuracy Comparison",
        color_continuous_scale="Blues"
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
        color="Category",
        title="Fake vs Real News Distribution"
    )

    st.plotly_chart(fig2,use_container_width=True)

    st.subheader("Dataset Category Distribution")

    categories = [
        "Fake","Bias","Conspiracy","Satire",
        "Hate","State","Junk Science","Reliable"
    ]

    counts = [12000,2000,500,800,300,200,100,3500]

    df_dist = pd.DataFrame({
        "Category":categories,
        "Count":counts
    })

    fig3 = px.pie(
        df_dist,
        names="Category",
        values="Count",
        title="News Category Distribution"
    )

    st.plotly_chart(fig3,use_container_width=True)

    st.subheader("Top News Categories")

    fig4 = px.bar(
        df_dist.sort_values("Count",ascending=False),
        x="Category",
        y="Count",
        color="Count",
        title="Top Categories in Dataset",
        color_continuous_scale="Plasma"
    )

    st.plotly_chart(fig4,use_container_width=True)

# ------------------- MODEL INFO -------------------

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

# Footer
st.sidebar.write("---")
st.sidebar.write("MSc Big Data Analytics Project")
