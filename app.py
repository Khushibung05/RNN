# ============================================
# BEAUTIFUL MODERN STREAMLIT UI
# WITH COLUMNS / CARDS / SECTIONS
# ============================================

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re
import string
import plotly.express as px

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
# MODERN CSS
# ============================================

st.markdown("""
<style>

/* Main Background */

.stApp{
    background: linear-gradient(
        135deg,
        #fff9c4,
        #e1f5fe,
        #fce4ec
    );
    background-attachment: fixed;
}

/* Remove Header */

header{
    visibility:hidden;
}

/* Main Container */

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

/* Main Title */

.main-title{
    text-align:center;
    font-size:60px;
    font-weight:800;
    color:#1e3a5f;
    margin-bottom:10px;
}

/* Subtitle */

.sub-title{
    text-align:center;
    font-size:28px;
    color:#546e7a;
    margin-bottom:40px;
}

/* Card Style */

.card{
    background:rgba(255,255,255,0.75);
    backdrop-filter:blur(10px);
    padding:28px;
    border-radius:25px;
    box-shadow:0px 6px 20px rgba(0,0,0,0.12);
    margin-bottom:25px;
}

/* Headings */

h1,h2,h3,h4{
    color:#1e3a5f !important;
}

/* Text */

p, li{
    color:#37474f;
    font-size:19px;
    line-height:1.8;
}

/* Text Area */

textarea{
    background:white !important;
    color:black !important;
    border-radius:18px !important;
    border:2px solid #90caf9 !important;
    font-size:20px !important;
    padding:15px !important;
}

/* Button */

.stButton > button{
    width:100%;
    height:60px;
    border:none;
    border-radius:18px;
    background:linear-gradient(
        135deg,
        #64b5f6,
        #ba68c8
    );
    color:white;
    font-size:24px;
    font-weight:bold;
    box-shadow:0px 4px 15px rgba(0,0,0,0.2);
}

/* Button Hover */

.stButton > button:hover{
    transform:scale(1.02);
    background:linear-gradient(
        135deg,
        #42a5f5,
        #ab47bc
    );
}

/* Info Boxes */

.metric-box{
    background:white;
    padding:20px;
    border-radius:18px;
    text-align:center;
    box-shadow:0px 4px 10px rgba(0,0,0,0.1);
}

/* Footer */

.footer{
    text-align:center;
    color:#455a64;
    font-size:18px;
    margin-top:30px;
}

</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD MODEL
# ============================================

model = load_model(
    "mental_health_rnn_model.h5",
    compile=False
)

# ============================================
# LOAD TOKENIZER
# ============================================

with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

# ============================================
# LOAD ENCODER
# ============================================

with open("label_encoder.pkl", "rb") as file:
    encoder = pickle.load(file)

# ============================================
# CONSTANTS
# ============================================

MAX_LENGTH = 50

# ============================================
# CLEAN TEXT
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
# HEADER
# ============================================

st.markdown("""
<div class="main-title">
🧠 AI-Based Mental Health Sentiment Monitoring System
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="sub-title">
Emotion Detection using Simple Recurrent Neural Networks
</div>
""", unsafe_allow_html=True)

# ============================================
# ABOUT SECTION USING 3 COLUMNS
# ============================================

st.markdown('<div class="card">',
unsafe_allow_html=True)

st.header("📘 About the Project")

col1, col2, col3 = st.columns(3)

with col1:

    st.subheader("🌟 Emotional AI")

    st.write("""
    - Detects emotional distress  
    - Supports early intervention  
    - Helps emotional wellness monitoring  
    """)

with col2:

    st.subheader("🤖 NLP Applications")

    st.write("""
    - Chatbots  
    - Sentiment Analysis  
    - Emotion Detection  
    - Text Classification  
    """)

with col3:

    st.subheader("🔁 Role of RNN")

    st.write("""
    RNN processes text sequentially
    and remembers previous words
    using hidden states for better
    understanding of emotions.
    """)

st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# INPUT + SAMPLE SECTION
# ============================================

left_col, right_col = st.columns([2,1])

# ============================================
# INPUT COLUMN
# ============================================

with left_col:

    st.markdown('<div class="card">',
    unsafe_allow_html=True)

    st.header("✍ Enter Your Thoughts")

    user_input = st.text_area(
        "",
        height=250,
        placeholder="Enter your thoughts or feelings here..."
    )

    st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# SAMPLE COLUMN
# ============================================

with right_col:

    st.markdown('<div class="card">',
    unsafe_allow_html=True)

    st.header("💭 Sample Inputs")

    st.info("""
I feel lonely and stressed lately
""")

    st.info("""
I am very happy today
""")

    st.info("""
Nobody understands me anymore
""")

    st.info("""
Life feels beautiful and peaceful
""")

    st.info("""
I feel mentally exhausted
""")

    st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# BUTTON
# ============================================

if st.button("🔍 Analyze Emotion"):

    if user_input.strip() == "":

        st.warning("Please enter some text.")

    else:

        sentiment, confidence, probabilities = predict_emotion(
            user_input
        )

        # ============================================
        # OUTPUT METRICS
        # ============================================

        st.markdown('<div class="card">',
        unsafe_allow_html=True)

        st.header("📊 Prediction Results")

        m1, m2, m3 = st.columns(3)

        with m1:

            st.markdown(f"""
            <div class="metric-box">
            <h3>Emotion</h3>
            <h2>{sentiment}</h2>
            </div>
            """, unsafe_allow_html=True)

        with m2:

            st.markdown(f"""
            <div class="metric-box">
            <h3>Confidence</h3>
            <h2>{confidence:.2f}%</h2>
            </div>
            """, unsafe_allow_html=True)

        with m3:

            if confidence >= 85:
                status = "High"

            elif confidence >= 60:
                status = "Moderate"

            else:
                status = "Low"

            st.markdown(f"""
            <div class="metric-box">
            <h3>Status</h3>
            <h2>{status}</h2>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # ============================================
        # GRAPH + GUIDANCE
        # ============================================

        graph_col, guide_col = st.columns([2,1])

        # ============================================
        # GRAPH SECTION
        # ============================================

        with graph_col:

            st.markdown('<div class="card">',
            unsafe_allow_html=True)

            st.header("📈 Emotion Confidence Graph")

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
                color="Emotion"
            )

            fig.update_traces(
                texttemplate='%{text:.2f}',
                textposition='outside'
            )

            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font_size=18
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.markdown('</div>', unsafe_allow_html=True)

        # ============================================
        # GUIDANCE SECTION
        # ============================================

        with guide_col:

            st.markdown('<div class="card">',
            unsafe_allow_html=True)

            st.header("💡 Wellness Tips")

            emotion = sentiment.lower()

            if "sad" in emotion \
            or "depression" in emotion \
            or "anxiety" in emotion \
            or "stress" in emotion:

                st.warning("""
🌿 Take a short break

🧘 Practice meditation

📞 Talk with someone you trust

🚶 Go for a short walk

🎵 Listen to calming music
""")

            elif "happy" in emotion \
            or "positive" in emotion:

                st.success("""
✨ Maintain positivity

🏃 Stay active

😊 Practice gratitude

🌟 Spread positivity
""")

            else:

                st.info("""
🌼 Maintain balance

🥗 Eat healthy food

💤 Sleep properly

🧘 Practice mindfulness
""")

            st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# FOOTER
# ============================================

st.markdown("""
<div class="footer">
Made with ❤️ using NLP, TensorFlow, SimpleRNN, and Streamlit
</div>
""", unsafe_allow_html=True)