# ============================================
# app.py
# AI-Based Mental Health Sentiment Monitoring
# Streamlit Deployment
# ============================================

# ============================================
# IMPORT LIBRARIES
# ============================================

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re
import string
import plotly.express as px
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Mental Health Sentiment Monitor",
    page_icon="🧠",
    layout="wide"
)

# ============================================
# CUSTOM CSS
# ============================================

st.markdown("""
<style>

/* Main App */

.stApp{
    background: linear-gradient(
        135deg,
        #141e30,
        #243b55
    );
}

/* Main Title */

.main-title{
    text-align:center;
    font-size:48px;
    font-weight:bold;
    color:white;
    margin-top:20px;
}

/* Subtitle */

.sub-title{
    text-align:center;
    font-size:22px;
    color:#dddddd;
    margin-bottom:30px;
}

/* Section Box */

.section-box{
    background: rgba(255,255,255,0.08);
    padding:25px;
    border-radius:20px;
    margin-bottom:25px;
    box-shadow:0px 4px 15px rgba(0,0,0,0.3);
    color:white;
}

/* Buttons */

.stButton > button{
    width:100%;
    background:#00c6ff;
    color:white;
    border:none;
    border-radius:10px;
    height:50px;
    font-size:20px;
    font-weight:bold;
}

.stButton > button:hover{
    background:#0072ff;
    color:white;
}

/* Text Area */

textarea{
    background-color:white !important;
    color:black !important;
    border-radius:12px !important;
    font-size:16px !important;
}

/* Success Box */

.stSuccess{
    border-radius:12px;
}

/* Warning Box */

.stWarning{
    border-radius:12px;
}

</style>
""", unsafe_allow_html=True)

# ============================================
# DEBUG CHECK
# ============================================

st.write("✅ App Started Successfully")

# ============================================
# LOAD MODEL
# ============================================

try:

    model = load_model(
        "mental_health_rnn_model.h5",
        compile=False
    )

    st.write("✅ Model Loaded")

except Exception as e:

    st.error(f"Model Loading Error: {e}")

# ============================================
# LOAD TOKENIZER
# ============================================

try:

    with open("tokenizer.pkl", "rb") as file:
        tokenizer = pickle.load(file)

    st.write("✅ Tokenizer Loaded")

except Exception as e:

    st.error(f"Tokenizer Error: {e}")

# ============================================
# LOAD LABEL ENCODER
# ============================================

try:

    with open("label_encoder.pkl", "rb") as file:
        encoder = pickle.load(file)

    st.write("✅ Label Encoder Loaded")

except Exception as e:

    st.error(f"Encoder Error: {e}")

# ============================================
# CONSTANTS
# ============================================

MAX_LENGTH = 50

# ============================================
# TEXT CLEANING FUNCTION
# ============================================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        f"[{re.escape(string.punctuation)}]",
        "",
        text
    )

    text = re.sub(r'\d+', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text

# ============================================
# PREDICTION FUNCTION
# ============================================

def predict_emotion(text):

    cleaned_text = clean_text(text)

    sequence = tokenizer.texts_to_sequences(
        [cleaned_text]
    )

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding='post'
    )

    prediction = model.predict(padded)

    predicted_class = np.argmax(prediction)

    sentiment = encoder.inverse_transform(
        [predicted_class]
    )[0]

    confidence = np.max(prediction) * 100

    return sentiment, confidence, prediction[0]

# ============================================
# HEADER SECTION
# ============================================

st.markdown(
    """
    <div class="main-title">
    🧠 AI-Based Mental Health Sentiment Monitoring System
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="sub-title">
    Emotion Detection using Simple Recurrent Neural Networks
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================
# ABOUT PROJECT
# ============================================

st.markdown('<div class="section-box">',
unsafe_allow_html=True)

st.header("📘 About the Project")

st.write("""
This AI-powered Mental Health Sentiment Monitoring System
uses Natural Language Processing (NLP)
and Simple Recurrent Neural Networks (RNN)
to analyze emotional sentiment from user text.

### Importance of Emotional AI
- Helps identify emotional distress
- Supports early emotional intervention
- Assists counselors and wellness systems

### NLP Applications
- Chatbots
- Sentiment Analysis
- Emotion Detection
- Text Classification

### Role of RNN
RNN processes text sequentially and remembers
previous words using hidden states,
allowing better understanding of sentence context.
""")

st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# USER INPUT SECTION
# ============================================

st.markdown('<div class="section-box">',
unsafe_allow_html=True)

st.header("✍ User Text Input")

st.write("### Sample Sentences")

st.code("""
• I feel lonely and stressed lately
• I am extremely happy today
• Nobody understands me anymore
• Life feels beautiful and peaceful
• I feel mentally exhausted
""")

user_input = st.text_area(
    "",
    height=180,
    placeholder="Enter your thoughts or feelings here..."
)

st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# PREDICTION BUTTON
# ============================================

if st.button("🔍 Analyze Emotion"):

    if user_input.strip() == "":

        st.warning("Please enter some text.")

    else:

        sentiment, confidence, probabilities = predict_emotion(
            user_input
        )

        # ============================================
        # OUTPUT SECTION
        # ============================================

        st.markdown('<div class="section-box">',
        unsafe_allow_html=True)

        st.header("📊 Prediction Output")

        st.success(
            f"Emotion Detected: {sentiment}"
        )

        st.info(
            f"Confidence Score: {confidence:.2f}%"
        )

        # Emotional Status

        if confidence >= 85:
            emotional_status = "High Confidence Detection"

        elif confidence >= 60:
            emotional_status = "Moderate Confidence Detection"

        else:
            emotional_status = "Low Confidence Detection"

        st.write(
            f"### Emotional Status: {emotional_status}"
        )

        st.markdown('</div>', unsafe_allow_html=True)

        # ============================================
        # VISUALIZATION SECTION
        # ============================================

        st.markdown('<div class="section-box">',
        unsafe_allow_html=True)

        st.header("📈 Emotion Probability Graph")

        labels = encoder.classes_

        prob_df = pd.DataFrame({
            "Emotion": labels,
            "Probability": probabilities
        })

        fig = px.bar(
            prob_df,
            x="Emotion",
            y="Probability",
            text="Probability",
            title="Emotion Confidence Distribution"
        )

        fig.update_traces(
            texttemplate='%{text:.2f}',
            textposition='outside'
        )

        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.markdown('</div>', unsafe_allow_html=True)

        # ============================================
        # EMOTIONAL GUIDANCE SECTION
        # ============================================

        st.markdown('<div class="section-box">',
        unsafe_allow_html=True)

        st.header("💡 Emotional Wellness Guidance")

        emotion = sentiment.lower()

        if "sad" in emotion \
        or "depression" in emotion \
        or "anxiety" in emotion \
        or "stress" in emotion:

            st.warning("""
🌿 Take a short break and relax.

🧘 Practice meditation or deep breathing.

📞 Talk with someone you trust.

🚶 Go for a short walk outdoors.

🎵 Listen to calming music.

💤 Ensure proper sleep and hydration.
""")

        elif "happy" in emotion \
        or "positive" in emotion:

            st.success("""
✨ Keep maintaining your positive mindset.

😊 Continue healthy habits.

🏃 Stay active and motivated.

📚 Practice gratitude daily.

🌟 Spread positivity around you.
""")

        else:

            st.info("""
🌼 Maintain a balanced lifestyle.

🥗 Eat healthy food.

💤 Get enough sleep.

🧘 Practice mindfulness regularly.
""")

        st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# FOOTER
# ============================================

st.markdown("""
<hr>

<center style="color:white;">
Made with ❤️ using NLP, TensorFlow, SimpleRNN, and Streamlit
</center>
""", unsafe_allow_html=True)