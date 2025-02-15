import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sumy.parsers.html import HtmlParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from sumy.summarizers.lsa import LsaSummarizer
import nltk

# Download the 'punkt' tokenizer models for nltk
nltk.download('punkt')

# App title
st.title("Bulk Meta Description Generator")

# Instructions for the user
st.markdown("""
### Instructions:
1. **Option 1**: Paste a list of URLs (one per line) in the text area below.
2. **Option 2**: Upload a CSV file containing a column named `Address` with the URLs.
3. Click the **Generate Meta Descriptions** button to generate meta descriptions for the URLs.
""")

# Input: Paste URLs or upload a CSV file
input_method = st.radio("Choose input method:", ("Paste URLs", "Upload CSV"))

urls = []

if input_method == "Paste URLs":
    # Text area for pasting URLs
    pasted_urls = st.text_area("Paste URLs (one per line):")
    if pasted_urls:
        urls = pasted_urls.strip().split("\n")
else:
    # File uploader for CSV
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'Address' in df.columns:
            urls = df['Address'].tolist()
        else:
            st.error("The uploaded CSV file must contain a column named 'Address'.")

# Button to extract meta descriptions
if st.button("Generate Meta Descriptions") and urls:
    results = []

    # Iterate over URLs
    for url in urls:
        try:
            # Parse the HTML content of the URL
            parser = HtmlParser.from_url(url, Tokenizer("english"))

            # Create a stemmer for the English language
            stemmer = Stemmer("english")

            # Create an LSA summarizer with the stemmer
            summarizer = LsaSummarizer(stemmer)

            # Set the stop words for the summarizer
            summarizer.stop_words = get_stop_words("english")

            # Generate a summary of the parsed document, limiting it to 3 sentences
            description = summarizer(parser.document, 3)

            # Combine the sentences into a single string
            description = " ".join([sentence._text for sentence in description])

            # If the description is longer than 155 characters, truncate it
            if len(description) > 155:
                description = description[:152] + '...'

            # Append the URL and its description to the results list
            results.append({
                'URL': url,
                'Meta Description': description
            })
        except Exception as e:
            st.error(f"Error processing {url}: {e}")

    # Display results in a table
    if results:
        st.success("Meta descriptions extracted successfully!")
        results_df = pd.DataFrame(results)
        st.dataframe(results_df)

        # Download button for results
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="meta_descriptions.csv",
            mime="text/csv"
        )
    else:
        st.warning("No meta descriptions were extracted.")
else:
    st.warning("Please provide URLs to extract meta descriptions.")
