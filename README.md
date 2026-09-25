# 📄 AI Resume Analyzer & Job Match System

An AI-powered web application that analyzes a resume against a job description and provides a semantic match score, matching keywords, and basic improvement suggestions.

## 🚀 Features

- 📑 Upload a resume in PDF format
- 📝 Paste a complete job description
- 🤖 Generate semantic embeddings using Sentence Transformers
- 📊 Calculate resume–job similarity using cosine similarity
- 🔑 Identify matching keywords
- 💡 Provide basic resume improvement suggestions
- 📄 View extracted resume text
- 🌐 Interactive Streamlit web interface

## 🛠️ Technologies Used

- **Python**
- **Streamlit**
- **Sentence Transformers**
- **NLP**
- **Cosine Similarity**
- **PyPDF**

## 🧠 How It Works

1. The user uploads a resume PDF.
2. The application extracts the text from the PDF.
3. The user enters a job description.
4. The resume and job description are converted into semantic embeddings using the `all-MiniLM-L6-v2` Sentence Transformer model.
5. Cosine similarity is used to calculate the match score.
6. The application identifies overlapping keywords.
7. Basic suggestions are displayed to help improve the resume.

## 📊 Example Output

The application provides:

- Resume Match Score
- Resume Character Count
- Number of Matching Keywords
- Matching Keywords
- Resume Improvement Suggestions
- Extracted Resume Text

## ▶️ How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/gopalshivani633-art/AI-Resume-Analyzer.git
