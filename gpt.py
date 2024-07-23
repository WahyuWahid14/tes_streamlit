import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
import os
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from wordcloud import WordCloud
from pathlib import Path


st.set_page_config(
    page_title="US Population Dashboard",
    page_icon="https://cdn.iconscout.com/icon/premium/png-512-thumb/training-development-3362447-2804086.png?f=webp&w=256",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

# Set up Streamlit
st.title('Sentiment Analysis Dashboard')
st.write('Analyze the sentiment of Indomie-related tweets')

# File uploader for the CSV file
uploaded_file = st.file_uploader("Choose a file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file, delimiter=";")
    st.write("Data Preview:")
    st.write(df.head())

    # Extract sentences
    sentences = df['full_text'].values

    # Load the pre-trained model and tokenizer
    pretrained = "mdhugol/indonesia-bert-sentiment-classification"
    model = AutoModelForSequenceClassification.from_pretrained(pretrained)
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    # Define sentiment labels
    label_index = {'LABEL_0': 'positive', 'LABEL_1': 'neutral', 'LABEL_2': 'negative'}

    # Function to analyze sentiment
    def analyze_sentiment(text):
        result = sentiment_analysis(text)
        label = label_index[result[0]['label']]
        score = result[0]['score']
        return pd.Series({'sentiment_label': label, 'sentiment_score': score, 'text': text})

    # Apply sentiment analysis to 'full_text' column
    result = df['full_text'].apply(analyze_sentiment)

    # Display the result
    st.write("Sentiment Analysis Result:")
    st.write(result.head())

    # Count the occurrences of each sentiment label
    sentiment_counts = result['sentiment_label'].value_counts()

    # Plot a bar chart using seaborn
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="viridis")
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment Label')
    plt.ylabel('Count')
    st.pyplot(plt)


    # Membuat barplot menggunakan Matplotlib
    fig, ax = plt.subplots()
    ax.bar(sentiment_counts.keys(), sentiment_counts.values(), color=['green', 'blue', 'red'])
    ax.set_xlabel('Kategori Sentimen')
    ax.set_ylabel('Jumlah')
    ax.set_title('Distribusi Sentimen')
    
else:
    st.write("Please upload a CSV file to proceed.")
