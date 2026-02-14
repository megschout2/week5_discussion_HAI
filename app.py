import os
import streamlit as st
import pandas as pd
from openai import OpenAI
from agent_panel import agent_panel

try:
    api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ["OPENAI_API_KEY"]

client = OpenAI(api_key=api_key)

st.set_page_config(page_title="Data Analysis Tool", layout="wide")
st.title("Interactive Data Analysis Tool")

df = pd.read_csv('movies.csv')

with st.sidebar:
    st.header("Data Filters")

    all_columns = df.columns.tolist()
    selected_columns = st.multiselect(
        "Select columns to include:",
        all_columns,
        default=all_columns
    )

    if not selected_columns:
        st.error("Please select at least one column.")
        st.stop()

    filtered_df = df[selected_columns]

    st.subheader("Row Filters")

    if 'Genre' in filtered_df.columns:
        genres = filtered_df['Genre'].dropna().unique()
        selected_genres = st.multiselect(
            "Filter by Genre:",
            genres,
            default=genres.tolist()
        )
        filtered_df = filtered_df[filtered_df['Genre'].isin(selected_genres)]

    if 'Release Year' in filtered_df.columns:
        min_year = int(filtered_df['Release Year'].min())
        max_year = int(filtered_df['Release Year'].max())
        year_range = st.slider(
            "Filter by Release Year:",
            min_year,
            max_year,
            (min_year, max_year)
        )
        filtered_df = filtered_df[
            (filtered_df['Release Year'] >= year_range[0]) &
            (filtered_df['Release Year'] <= year_range[1])
        ]

    if 'IMDB Rating' in filtered_df.columns:
        min_rating = float(filtered_df['IMDB Rating'].min())
        max_rating = float(filtered_df['IMDB Rating'].max())
        rating_range = st.slider(
            "Filter by IMDB Rating:",
            min_rating,
            max_rating,
            (min_rating, max_rating)
        )
        filtered_df = filtered_df[
            (filtered_df['IMDB Rating'] >= rating_range[0]) &
            (filtered_df['IMDB Rating'] <= rating_range[1])
        ]

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Filtered Dataset")
    st.write(filtered_df)

    st.subheader("Ask a Question")
    user_question = st.text_input(
        "What would you like to know about this data?",
        placeholder="e.g., What is the average IMDB rating?"
    )
    show_chart = st.checkbox("Show chart")
    analyze_button = st.button("Analyze", type="primary")

with col2:
    agent_panel(client, analyze_button, user_question, filtered_df, show_chart)
