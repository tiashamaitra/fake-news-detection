import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Define clickbait keywords
clickbait_keywords = [
    'shocking', 'exclusive', 'breaking', 'urgent', 'warning',
    'incredible', 'unbelievable', 'miracle', 'secret', 'banned',
    'revolutionary', 'mind-blowing', 'you won\'t believe',
    'sensational', 'amazing', 'stunning', 'viral', 'exposed',
    'jaw-dropping', 'must see', 'conspiracy', 'scandal'
]

def calculate_keyword_density(text, keywords):
    if pd.isna(text):
        return 0
    text = text.lower()
    word_count = len(text.split())
    if word_count == 0:
        return 0
    keyword_count = sum(text.count(keyword.lower()) for keyword in keywords)
    return keyword_count / word_count

class LengthBasedFakeNewsDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
    @classmethod
    def load_model(cls, filepath):
        with open(filepath, 'rb') as f:
            model_components = pickle.load(f)
        detector = cls()
        detector.model = model_components['model']
        detector.scaler = model_components['scaler']
        return detector
        
    def extract_length_features(self, df):
        features = pd.DataFrame()
        features['title_char_count'] = df['title'].str.len()
        features['title_word_count'] = df['title'].str.split().str.len()
        features['title_avg_word_length'] = features['title_char_count'] / features['title_word_count']
        features['text_char_count'] = df['text'].str.len()
        features['text_word_count'] = df['text'].str.split().str.len()
        features['text_avg_word_length'] = features['text_char_count'] / features['text_word_count']
        features['text_to_title_ratio'] = features['text_word_count'] / features['title_word_count']
        features['char_to_word_ratio'] = features['text_char_count'] / features['text_word_count']
        return features
        
    def predict(self, test_data):
        X = self.extract_length_features(test_data)
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        return predictions, probabilities

class IntegratedFakeNewsModel:
    def __init__(self):
        self.sentiment_model = None
        self.sentiment_vectorizer = None
        self.keyword_model = None
        self.tfidf_title_vectorizer = None
        self.tfidf_text_vectorizer = None
        self.linguistic_model = None
        self.linguistic_vectorizer = None
        self.length_based_detector = None

    def create_keyword_features(self, df):
        features = pd.DataFrame(index=df.index)
        features['title_keyword_density'] = df['title'].apply(
            lambda x: calculate_keyword_density(x, clickbait_keywords)
        )
        features['text_keyword_density'] = df['text'].apply(
            lambda x: calculate_keyword_density(x, clickbait_keywords)
        )
        
        title_features = self.tfidf_title_vectorizer.transform(df['title'].fillna(''))
        text_features = self.tfidf_text_vectorizer.transform(df['text'].fillna(''))
        
        title_df = pd.DataFrame(
            title_features.toarray(),
            columns=[f'title_tfidf_{i}' for i in range(title_features.shape[1])],
            index=df.index
        )
        
        text_df = pd.DataFrame(
            text_features.toarray(),
            columns=[f'text_tfidf_{i}' for i in range(text_features.shape[1])],
            index=df.index
        )
        
        features = pd.concat([features, title_df, text_df], axis=1)
        return features

    def load_all_models(self):
        # Load sentiment models
        with open(r'random_forest_model.pkl', 'rb') as f:
            self.sentiment_model = pickle.load(f)
        with open(r'tfidf_vectorizer.pkl', 'rb') as f:
            self.sentiment_vectorizer = pickle.load(f)

        # Load keyword models
        with open(r'rf_classifier.pkl', 'rb') as f:
            self.keyword_model = pickle.load(f)
        with open(r'tfidf_title_vectorizer.pkl', 'rb') as f:
            self.tfidf_title_vectorizer = pickle.load(f)
        with open(r'tfidf_text_vectorizer.pkl', 'rb') as f:
            self.tfidf_text_vectorizer = pickle.load(f)

        # Load linguistic models
        self.linguistic_model = joblib.load(r'fake_news_model_updated.pkl')
        self.linguistic_vectorizer = joblib.load(r'tfidf_vectorizer_updated.pkl')

        # Load length-based detector
        self.length_based_detector = LengthBasedFakeNewsDetector.load_model(r"my_model.pkl")

    def predict(self, input_data):
        # Generate features for sentiment analysis
        sentiment_features = self.sentiment_vectorizer.transform(input_data['text'])
        sentiment_predictions = self.sentiment_model.predict_proba(sentiment_features)[:, 1]
        
        # Generate features for keyword density analysis
        keyword_features = self.create_keyword_features(input_data)
        keyword_predictions = self.keyword_model.predict_proba(keyword_features)[:, 1]
        
        # Generate features for linguistic model
        linguistic_features = self.linguistic_vectorizer.transform(input_data['text'])
        linguistic_predictions = self.linguistic_model.predict_proba(linguistic_features)[:, 1]
        
        # Generate features for length-based model
        _, length_probabilities = self.length_based_detector.predict(input_data)
        
        # Combine predictions with weighted average
        weight_sentiment = 0.3
        weight_keyword = 0.3
        weight_linguistic = 0.2
        weight_length = 0.2
        
        combined_predictions = (
            weight_sentiment * sentiment_predictions +
            weight_keyword * keyword_predictions +
            weight_linguistic * linguistic_predictions +
            weight_length * length_probabilities
        )
        
        return combined_predictions[0]
