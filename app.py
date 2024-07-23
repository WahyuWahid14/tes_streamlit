import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
import streamlit as st

import os
import collections
# %matplotlib inline
sns.set(color_codes=True)

# Melihat Data

def df(file_path, delimiter=";"):
    df = pd.read_csv(file_path, delimiter=delimiter)
    return df
    
    
data = "/home/wahyu/Documents/Project Data Analyst/Code/tweet_indomie.csv"
df = df(data)

print("\n")


from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

pretrained= "mdhugol/indonesia-bert-sentiment-classification"

model = AutoModelForSequenceClassification.from_pretrained(pretrained)
tokenizer = AutoTokenizer.from_pretrained(pretrained)

sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

label_index = {'LABEL_0': 'positive', 'LABEL_1': 'neutral', 'LABEL_2': 'negative'}

def analyze_sentiment(text):
  result = sentiment_analysis(text)
  label = label_index[result[0]['label']]
  score = result[0]['score']
  return pd.Series({'sentiment_label': label, 'sentiment_score': score, 'text': text})

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
st.dataframe(data)

# Grafik
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="viridis")
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment Label')
plt.ylabel('Count')

# Create a tilted bar chart using Altair
chart = alt.Chart(sentiment_counts).mark_bar().encode(
    x=alt.X('sentiment_label', sort='-y', axis=alt.Axis(labelAngle=-45)),
    y='count',
    color='sentiment_label'
).properties(
    width=600,
    height=400
).configure_axis(
    labelFontSize=12,
    titleFontSize=14
)

# Display the chart in Streamlit
st.altair_chart(chart)

print("\n")
print("code beralan dengan lancar")


# # # Apply sentiment analysis to 'cleaned_text' column
# def apply_sentiment_analysis(df, column_name):                                   
#   df[f'{column_name}_sentiment'] = df[column_name].apply(analyze_sentiment)
#   return df


# # # Count the occurrences of each sentiment label
# def count_sentiment_labels(df, column_name):
#   sentiment_counts = df[column_name].value_counts()
#   return sentiment_counts


# def plot_sentiment_distribution(sentiment_counts):
#     """
#     Membuat plot bar chart dari distribusi sentimen menggunakan seaborn.

#     """
#     sns.set(style="whitegrid")
#     plt.figure(figsize=(8, 6))
#     sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="viridis")
#     plt.title('Sentiment Distribution')
#     plt.xlabel('Sentiment Label')
#     plt.ylabel('Count')