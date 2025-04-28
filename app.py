from dotenv import load_dotenv
import streamlit as st
import re
import requests
from bs4 import BeautifulSoup
from wordcloud import WordCloud, STOPWORDS
import textstat
import matplotlib.pyplot as plt
from core import (
    import_google_api,
    embedding_function,
    unpickle,
    create_collection,
    unzipping_the_dataset,
    persistent_client,
    get_article,
    summarize_article,
    get_full_article
)

# Streamlit App Configuration - This must be first
st.set_page_config(page_title="Stanford Encyclopedia of Philosophy chatbot", page_icon="🏛️", layout="wide")

# -----------------------------------
# Importing Google API key, embedding function, collection, and retry
# -----------------------------------
client = import_google_api()
gemini_embedding_function = embedding_function(client)
embed_fn, collection = persistent_client(gemini_embedding_function)

# -----------------------------------
# Streamlit UI
# -----------------------------------

def estimate_reading_stats(text, education_level='college'):
    word_count = len(text.split())

    # Average reading speeds (words per minute) by education level
    speeds = {
        'elementary': 125,
        'high_school': 225,
        'college': 275,
        'graduate': 325,
        'expert': 450
    }

    # Get words per minute for the given education level
    wpm = speeds.get(education_level.lower(), 275)
    minutes = round(word_count / wpm, 2)

    # Estimate difficulty using Flesch Reading Ease (0–100)
    # Higher = easier
    difficulty_score = textstat.flesch_reading_ease(text)

    # Interpret the score
    if difficulty_score >= 90:
        difficulty = "Very Easy (5th grade)"
    elif difficulty_score >= 80:
        difficulty = "Easy (6th grade)"
    elif difficulty_score >= 70:
        difficulty = "Fairly Easy (7th grade)"
    elif difficulty_score >= 60:
        difficulty = "Standard (8th–9th grade)"
    elif difficulty_score >= 50:
        difficulty = "Fairly Difficult (10th–12th grade)"
    elif difficulty_score >= 30:
        difficulty = "Difficult (College)"
    else:
        difficulty = "Very Difficult (College Graduate+)"

    return {
        'word_count': word_count,
        'estimated_minutes': minutes,
        'difficulty': difficulty
    }

# Example usage
text = "This is a sample passage to evaluate how long it might take someone to read it based on their educational background."
stats = estimate_reading_stats(text, 'high_school')
print(stats)

def main():
    unzipping_the_dataset()
    
    #Title page
    # Create two columns: one small for the icon, one large for the title
    col1, col2 = st.columns([1, 18])

    with col1:
        st.image("img/sep-man-red.png", width=40)

    with col2:
        st.title("Stanford Encyclopedia of Philosophy chatbot")  # 👈 title next to logo

    st.markdown("Search for encyclopedia articles, analyze and visualize it, or generate a professional summary.")

    # Sidebar for settings
    with st.sidebar:
        st.markdown("### About")
        st.markdown("This app explains Stanford Encyclopedia of Philosophy (SEP) articles, summarizes and visualizes them. Powered by Google Gemini.")
        st.markdown("[Click here to visit the SEP](https://plato.stanford.edu/)")
        st.markdown("### Made By")
        st.markdown("Bruno M. with ❤️")
        col1, col2 = st.columns([1, 10])
        with col1:
            st.image("img/linkedin.svg", width=20)
        with col2:
            st.markdown("[Connect with me on LinkedIn](https://www.linkedin.com/in/bruno-m-1141262b3/)")

    # Tabs for Search, Analysis, Visualization, and Summarization
    explain_tab, visualize_tab, summarize_tab = st.tabs(["🤖 Explain entry", "📈 Visualize entry", "📝 Summarize entry"])

    with explain_tab:
        with st.form(key="query_form"):
            user_query = st.text_input("💬 Query the encyclopedia:", placeholder="e.g., Explain Abelard's logic.")
            submit_button = st.form_submit_button("🔎 Search the encyclopedia")

        if submit_button and user_query:
            with st.spinner("Searching for the article..."):
                article = get_article(user_query, embed_fn, collection, client)
                sep_full_article = get_full_article(user_query, embed_fn, collection, client)

            # Save into session_state
            st.session_state['article'] = article
            st.session_state['user_query'] = user_query
            st.session_state['sep_full_article'] = sep_full_article

            # Clean and save cleaned text
            cleaned_text = re.sub(r'\s+', ' ', article.replace('\n', ' ')).strip()
            st.session_state['cleaned_text'] = cleaned_text

            # Clean and save cleaned text of the sep article
            cleaned_text_full = sep_full_article
            st.session_state['cleaned_text_full'] = cleaned_text_full

        # Always display article if it exists in session state
        if 'article' in st.session_state:
            st.subheader("📖 Answer")
            st.markdown(st.session_state['article'])

    with visualize_tab:
        # Dropdown for selecting education level
        education_level = st.selectbox(
        "Select Reader's Education Level:",
        ('elementary', 'high_school', 'college', 'graduate', 'expert'),
        index=2  # 2 = college
    )

        # Read cleaned_text and cleaned_text_full from session_state
        cleaned_text = st.session_state.get('cleaned_text', '')
        cleaned_text_full = st.session_state.get('cleaned_text_full', '')

        # Check if cleaned_text is available
        if cleaned_text:
            reading_estimate = estimate_reading_stats(cleaned_text, education_level=education_level)

            st.markdown("##### Reading statistics of entry")
            st.write(f"**Word Count:** {reading_estimate['word_count']}")
            st.write(f"**Estimated Reading Time:** {reading_estimate['estimated_minutes']} minutes")
            st.write(f"**Difficulty:** {reading_estimate['difficulty']}")

        # Check if cleaned_text_full is available
        if cleaned_text_full:
            reading_estimate_full = estimate_reading_stats(cleaned_text_full, education_level=education_level)

            st.markdown("##### Reading statistics of full entry")
            st.write(f"**Word Count:** {reading_estimate_full['word_count']}")
            st.write(f"**Estimated Reading Time:** {reading_estimate_full['estimated_minutes']} minutes")
            st.write(f"**Difficulty:** {reading_estimate_full['difficulty']}")

        # If neither cleaned_text nor cleaned_text_full is available, show an error message
        if not cleaned_text and not cleaned_text_full:
            st.error("🚨 Please input an entry in the `Explain Entry` tab first!")

        # Only show the word cloud if there is valid text (cleaned_text or cleaned_text_full)
        if cleaned_text and cleaned_text_full:
            # ----- Word Cloud -----        
            st.subheader("☁️ Word Cloud")
            custom_stopwords = set(STOPWORDS)
            custom_stopwords.update([
                "said", "also", "one", "would", "could", "us", "get", "like"
            ])

            # Check if word cloud exists in session state and matches current cleaned_text
            if 'wordcloud' not in st.session_state or st.session_state.get('wordcloud_text') != cleaned_text:
                # Generate word cloud with a fixed random seed
                wordcloud = WordCloud(
                    width=800,
                    height=400,
                    background_color='white',
                    colormap='viridis',
                    stopwords=custom_stopwords,
                    random_state=42  # Fixed seed for consistent word placement
                ).generate(cleaned_text or cleaned_text_full)  # Use whichever text is available

                # Store word cloud and text in session state
                st.session_state['wordcloud'] = wordcloud
                st.session_state['wordcloud_text'] = cleaned_text or cleaned_text_full

            # Retrieve word cloud from session state
            wordcloud = st.session_state['wordcloud']

            # Plotting the word cloud
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)


    with summarize_tab:
        article = st.session_state.get('article')
        user_query = st.session_state.get('user_query')

        if article and user_query:
            # Check if article or query has changed
            if 'summed_article' in st.session_state:
                prev_article = st.session_state.get('prev_article')
                prev_query = st.session_state.get('prev_query')
                if prev_article != article or prev_query != user_query:
                    st.session_state.pop('summed_article', None)  # Clear previous summary

            # Generate new summary if none exists
            if 'summed_article' not in st.session_state:
                with st.spinner("Generating summary for the article..."):
                    summed_article = summarize_article(user_query, embed_fn, collection, client)
                    st.session_state['summed_article'] = summed_article
                    # Store current inputs for future comparison
                    st.session_state['prev_article'] = article
                    st.session_state['prev_query'] = user_query

            st.subheader("📖 Summary")
            st.markdown(st.session_state['summed_article'])
        else:
            st.error("🚨 Please input an entry in the `Explain Entry` tab first!")
            st.session_state.pop('summed_article', None)  # Clear stale summary


if __name__ == "__main__":
    main()
