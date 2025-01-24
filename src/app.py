import streamlit as st
from model import (create_conversational_rag_chain, get_session_history, load_pdf_documents)
import pandas as pd

# Streamlit UI
st.set_page_config(page_title="Document QA Bot", layout="wide")
st.markdown("<h1 style='color: #1E3D59;'>Document QA Bot</h1>", unsafe_allow_html=True)
st.write("Upload PDFs and ask questions about any content - text, numbers, or tables.")

# Initialize session state
if "store" not in st.session_state:
    st.session_state.store = {}
    st.session_state.uploaded_files = []

# File upload section
with st.sidebar:
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True,
        help="Upload one or more PDF files containing text, numbers, or tables"
    )

if uploaded_files:
    try:
        # Process uploaded documents
        with st.spinner("Processing documents..."):
            documents = load_pdf_documents(uploaded_files)
            vectorstore, retriever, extracted_data = documents["vectorstore"], documents["retriever"], documents["extracted_data"]

        # Display tables in an expandable section
        if documents.get("tables"):
            with st.expander("View Extracted Tables", expanded=False):
                for table in documents["tables"]:
                    if table.get("raw_data"):
                        st.markdown(f"**Table from Page {table['page']}**")
                        df = pd.DataFrame(table["raw_data"])
                        st.dataframe(df, use_container_width=True)
                        st.markdown("---")

        # Display extracted data in expandable sections
        if not extracted_data["text_content"].empty:
            with st.expander("View Document Content", expanded=False):
                for _, row in extracted_data["text_content"].iterrows():
                    st.markdown(f"**Page {row['page']}** (Source: {row['source']})")
                    st.text(row['content'])
                    st.markdown("---")

        if not extracted_data["numerical_data"].empty:
            with st.expander("View Extracted Numerical Data", expanded=False):
                st.dataframe(
                    extracted_data["numerical_data"],
                    use_container_width=True,
                    hide_index=True
                )

        # Main interaction area
        st.subheader("Ask Questions")
        col1, col2 = st.columns([2, 1])
        
        with col2:
            session_id = st.text_input(
                "Session ID",
                value="default_session",
                help="Enter a unique session ID to maintain conversation history"
            ).strip() or "default_session"

        # Initialize RAG chain
        conversational_rag_chain = create_conversational_rag_chain(retriever)

        with col1:
            user_input = st.text_input(
                "Ask a question:",
                placeholder="E.g., 'What are the key points mentioned in the document?' or 'Can you find any numerical data about [topic]?'"
            )

        if user_input:
            if not documents["vectorstore"].docstore._dict:  # Check if there's actual content
                st.warning("No document content was found to analyze. Please make sure your PDF contains readable text.")
            else:
                with st.spinner("Analyzing..."):
                    session_history = get_session_history(session_id)
                    response = conversational_rag_chain.invoke(
                        {"input": user_input},
                        {"configurable": {"default": session_id}}
                    )
                    
                # Display response in a clean format
                st.markdown("### Response:")
                st.markdown(f"_{response['answer']}_")
                
                # Show chat history in an expandable section
                with st.expander("View Conversation History", expanded=False):
                    for msg in session_history.messages:
                        st.markdown(f"**{'User' if msg.type == 'human' else 'Assistant'}:** {msg.content}")

    except Exception as e:
        st.error(f"""
        Error processing the PDF document(s). This could be due to:
        - The PDF contains no readable text or tables
        - The PDF format is not supported
        - The document is password protected
        
        Error details: {str(e)}
        
        Please try uploading a different PDF document.
        """)

else:
    # Display instructions when no files are uploaded
    st.info("""
    👋 Welcome to the Document QA Bot!
    
    To get started:
    1. Upload any PDF document using the sidebar
    2. View the extracted data in the expandable section
    3. Ask questions about any content in the document
    
    Example questions:
    - What are the key points mentioned in the document?
    - Can you find any numerical data about [topic]?
    - What does the document say about [specific subject]?
    - Are there any trends or patterns in the data?
    
    Note: Make sure your PDF contains readable text content.
    """)
