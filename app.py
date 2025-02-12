
import streamlit as st
import pandas as pd
from model import IntegratedFakeNewsModel

# Streamlit interface setup
st.set_page_config(page_title="Fake News Detector", layout="centered")

# Custom styling
st.markdown("""
    <style>
    .main-header {
        font-size: 40px;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 30px;
        padding: 20px;
    }
    .result-real {
        padding: 20px;
        background-color: #90EE90;
        border-radius: 10px;
        text-align: center;
        color: #006400;
        font-size: 24px;
        margin: 20px 0;
    }
    .result-fake {
        padding: 20px;
        background-color: #FFB6C1;
        border-radius: 10px;
        text-align: center;
        color: #8B0000;
        font-size: 24px;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    model = IntegratedFakeNewsModel()
    model.load_all_models()
    return model

try:
    model = load_model()
    st.markdown("<h1 class='main-header'>📰 Fake News Detector</h1>", unsafe_allow_html=True)
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Input fields
title = st.text_area("News Title", placeholder="Enter the news title here...", height=100)
content = st.text_area("News Content", placeholder="Enter the news content here...", height=200)

# Predict button
if st.button("Predict", type="primary"):
    if title.strip() and content.strip():
        try:
            # Prepare input data
            input_data = pd.DataFrame({
                'title': [title],
                'text': [content]
            })
            
            # Get prediction probability
            probability = model.predict(input_data)
            
            # Show prediction result
            if probability > 0.5:
                st.markdown(
                    f"""
                    <div class='result-real'>
                        ✅ REAL NEWS<br>
                        Confidence: {probability:.2%}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class='result-fake'>
                        ❌ FAKE NEWS<br>
                        Confidence: {(1-probability):.2%}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            # Display additional metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Title Length", len(title))
            with col2:
                st.metric("Content Length", len(content))
                
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
    else:
        st.warning("Please enter both title and content")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666666; font-size: 14px;'>"
    "Fake News Detection System powered by Machine Learning</p>", 
    unsafe_allow_html=True
)
