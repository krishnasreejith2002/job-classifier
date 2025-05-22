import streamlit as st
import pandas as pd
import joblib
from your_scraper_module import scrape_karkidi_jobs, clean_skills, classify_new_jobs

# Load saved vectorizer and model once
@st.cache_resource
def load_models():
    model = joblib.load("karkidi_kmeans.pkl")
    vectorizer = joblib.load("karkidi_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_models()

st.title("Karkidi Job Scraper & Cluster Classifier")

keyword = st.text_input("Enter job keyword (e.g., data science):", value="data science")
pages = st.slider("Number of pages to scrape:", 1, 5, 1)

if st.button("Scrape & Classify Jobs"):
    with st.spinner("Scraping jobs... This may take a moment"):
        jobs_df = scrape_karkidi_jobs(keyword=keyword, pages=pages)
    st.success(f"Scraped {len(jobs_df)} jobs")

    st.write("Classifying jobs into clusters...")
    classified_df = classify_new_jobs(jobs_df, vectorizer, model)
    st.write(classified_df[["Title", "Company", "Location", "Skills", "Cluster"]])

    cluster_choice = st.selectbox("Filter jobs by cluster:", options=sorted(classified_df["Cluster"].unique()))
    filtered_jobs = classified_df[classified_df["Cluster"] == cluster_choice]
    
    st.write(f"Jobs in cluster {cluster_choice}:")
    st.dataframe(filtered_jobs[["Title", "Company", "Location", "Skills"]])

    if st.button("Download CSV"):
        csv = filtered_jobs.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download CSV", data=csv, file_name="filtered_jobs.csv", mime="text/csv")
