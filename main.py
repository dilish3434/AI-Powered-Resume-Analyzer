
import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(path):
    doc = fitz.open(stream=path.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_skills(text):
    doc = nlp(text)
    skills = [ent.text.lower() for ent in doc.ents if ent.label_ in ["ORG", "PERSON", "GPE"]]
    return list(set(skills))

def match_jobs(resume_text, jobs_df):
    tfidf = TfidfVectorizer(stop_words='english')
    corpus = [resume_text] + jobs_df['description'].tolist()
    vectors = tfidf.fit_transform(corpus)
    similarity = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    jobs_df['match_score'] = similarity
    return jobs_df.sort_values(by='match_score', ascending=False).head(5)

st.title("🚀 AI Resume Analyzer & Job Matcher")
uploaded_file = st.file_uploader("📄 Upload your Resume (PDF format)", type="pdf")

if uploaded_file:
    resume_text = extract_text_from_pdf(uploaded_file)
    st.subheader("📜 Extracted Resume Text")
    st.write(resume_text[:800] + "..." if len(resume_text) > 800 else resume_text)

    skills = extract_skills(resume_text)
    st.subheader("🧠 Extracted Skills (NER-based)")
    st.write(skills)

    jobs_df = pd.read_csv("data/jobs.csv")
    matches = match_jobs(resume_text, jobs_df)

    st.subheader("💼 Top Job Matches")
    st.dataframe(matches[['title', 'company', 'match_score']])
