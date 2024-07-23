import streamlit as st 
import matplotlib as plt
import seaborn as sns
import altair as alt
from app import analyze_sentiment
from app import df
from app import apply_sentiment_analysis
from app import count_sentiment_labels


# Apply sentiment analysis to 'cleaned_text' column
result = df['full_text'].apply(analyze_sentiment)

# Count the occurrences of each sentiment label
sentiment_counts = result['sentiment_label'].value_counts()


st.set_page_config(
    page_title="US Population Dashboard",
    page_icon="https://cdn.iconscout.com/icon/premium/png-512-thumb/training-development-3362447-2804086.png?f=webp&w=256",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")


# Judul dan Teks
st.title('Dashboard Sederhana dengan Streamlit')
st.header('Ini adalah header')
st.text('Ini adalah teks biasa yang menampilkan beberapa informasi.')

# Data
# Apply sentiment analysis to 'cleaned_text' column
data = df['full_text'].apply(analyze_sentiment)
# Count the occurrences of each sentiment label
sentiment_counts = data['sentiment_label'].value_counts()

# Menampilkan Data
st.subheader('Tabel Data dengan DataFrame')
# st.dataframe(data)

# Grafik
st.subheader('DISTRIBUTION SENTIMENT')
# sns.set(style="whitegrid")
# plt.figure(figsize=(8, 6))
# sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="viridis")
# plt.title('Sentiment Distribution')
# plt.xlabel('Sentiment Label')
# plt.ylabel('Count')

