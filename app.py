import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

# -----------------------------
# Load AI model
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()


# -----------------------------
# Extract text from PDF
# -----------------------------
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# -----------------------------
# Calculate resume-job match
# -----------------------------
def calculate_match(resume_text, job_description):
    resume_embedding = model.encode(
        resume_text,
        convert_to_tensor=True
    )

    job_embedding = model.encode(
        job_description,
        convert_to_tensor=True
    )

    similarity = util.cos_sim(
        resume_embedding,
        job_embedding
    )

    score = float(similarity[0][0]) * 100

    return round(score, 2)


# -----------------------------
# Find important keywords
# -----------------------------
def find_keywords(resume_text, job_description):
    resume_words = set(
        word.lower().strip(".,:;()[]{}")
        for word in resume_text.split()
    )

    job_words = set(
        word.lower().strip(".,:;()[]{}")
        for word in job_description.split()
    )

    common_words = resume_words.intersection(job_words)

    ignored_words = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "are",
        "you",
        "your",
        "from",
        "have",
        "will",
        "our",
        "their",
        "they",
        "using",
        "into",
        "about",
        "work",
        "working"
    }

    keywords = [
        word for word in common_words
        if len(word) > 2 and word not in ignored_words
    ]

    return sorted(keywords)


# -----------------------------
# User Interface
# -----------------------------
st.title("📄 AI Resume Analyzer")

st.write(
    "Upload your resume and paste a job description "
    "to analyze how well your resume matches the role."
)

st.divider()

# Resume upload
uploaded_resume = st.file_uploader(
    "Upload your resume (PDF)",
    type=["pdf"]
)

# Job description
job_description = st.text_area(
    "Paste the job description",
    height=250,
    placeholder="Paste the complete job description here..."
)

# Analyze button
if st.button("🔍 Analyze Resume", type="primary"):

    if uploaded_resume is None:
        st.warning("Please upload your resume PDF.")

    elif not job_description.strip():
        st.warning("Please paste the job description.")

    else:

        with st.spinner("Analyzing your resume..."):

            resume_text = extract_text_from_pdf(
                uploaded_resume
            )

            if not resume_text.strip():
                st.error(
                    "Could not extract text from the PDF. "
                    "Please upload a text-based PDF."
                )

            else:

                score = calculate_match(
                    resume_text,
                    job_description
                )

                keywords = find_keywords(
                    resume_text,
                    job_description
                )

                st.success("Analysis completed!")

                # Score
                st.subheader("📊 Resume Match Score")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Match Score",
                        f"{score}%"
                    )

                with col2:
                    st.metric(
                        "Resume Characters",
                        len(resume_text)
                    )

                with col3:
                    st.metric(
                        "Matching Keywords",
                        len(keywords)
                    )

                st.progress(
                    min(score / 100, 1.0)
                )

                # Keywords
                st.subheader("🔑 Matching Keywords")

                if keywords:
                    st.write(
                        ", ".join(keywords[:40])
                    )
                else:
                    st.write(
                        "No significant matching keywords found."
                    )

                # Resume preview
                with st.expander("📄 View Extracted Resume Text"):
                    st.text(resume_text[:10000])

                # Basic recommendations
                st.subheader("💡 Suggestions")

                if score < 40:
                    st.write(
                        "• Add more skills and keywords that "
                        "are directly relevant to the job description."
                    )
                    st.write(
                        "• Highlight projects related to the role."
                    )

                elif score < 70:
                    st.write(
                        "• Your resume has some relevant content. "
                        "Consider adding more role-specific skills."
                    )
                    st.write(
                        "• Quantify your project achievements "
                        "where possible."
                    )

                else:
                    st.write(
                        "• Your resume contains strong overlap "
                        "with the job description."
                    )
                    st.write(
                        "• Continue tailoring your resume to "
                        "the specific position."
                    )