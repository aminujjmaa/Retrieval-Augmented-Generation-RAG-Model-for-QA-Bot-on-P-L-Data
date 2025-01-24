# Conversational RAG with PDF Uploads and Chat History
This project implements a Conversational Retrieval-Augmented Generation (RAG) system using PDFs as the source of information. The system allows users to upload PDFs, chat with the content of the PDFs, and maintains a session history for continuous conversations.

# Features:
Upload multiple PDF files and extract their content.
Engage in a conversation with the assistant based on the uploaded PDFs.
Session-based chat history management.
Polite responses for queries that don't relate to the uploaded PDFs.
Prerequisites
Before setting up the project, ensure you have the following installed on your system:

Python (preferably 3.8 or later)
pip (Python package installer)

# Setup Instructions


## Installation

clone project repo using

git clone <Assignment>

# create env file
python -m venv env

# Activate the virtual environment:
venv\Scripts\activate

# Install dependencies:
pip install -r requirements.txt

# Environment variables:
groq_api=<your_groq_api_key>

# Run the application:
streamlit run app.py




    