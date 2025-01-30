import streamlit as st
import os
from model import (create_conversational_rag_chain, get_session_history, load_documents)
import pandas as pd

# Configure Streamlit for larger file uploads
st.set_page_config(
    page_title="Document QA Bot",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Increase memory limits for file processing
if not os.environ.get("STREAMLIT_SERVER_MAX_UPLOAD_SIZE"):
    os.environ["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] = "2048"  # Size in MB (2GB)

# Increase timeout for longer processing
if not os.environ.get("STREAMLIT_SERVER_TIMEOUT"):
    os.environ["STREAMLIT_SERVER_TIMEOUT"] = "300"  # Timeout in seconds

# Streamlit UI
st.markdown("<h1 style='color: #1E3D59;'>Document QA Bot</h1>", unsafe_allow_html=True)
st.write("Upload documents and ask questions about any content - text, tables, or images.")

# Initialize session state
if "store" not in st.session_state:
    st.session_state.store = {}
    st.session_state.uploaded_files = []

# File upload section
with st.sidebar:
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "xlsx", "xls", "ppt", "pptx", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="Upload documents in PDF, Excel, PowerPoint, or image formats"
    )

if uploaded_files:
    try:
        # Process uploaded documents
        with st.spinner("Processing documents..."):
            documents = load_documents(uploaded_files)
            vectorstore, retriever, extracted_data = documents["vectorstore"], documents["retriever"], documents["extracted_data"]

        # Display processing errors if any
        if documents.get("processing_errors"):
            st.warning("Some files had processing issues:")
            for error in documents["processing_errors"]:
                st.error(error)

        # Display tables in an expandable section
        if documents.get("tables"):
            with st.expander("View Extracted Tables", expanded=False):
                for table in documents["tables"]:
                    if table.get("raw_data"):
                        st.markdown(f"**Table from {table['source']} (Page/Sheet {table.get('page', 'N/A')})**")
                        df = pd.DataFrame(table["raw_data"])
                        st.dataframe(df, use_container_width=True)
                        st.markdown("---")

        # Display images if present
        if documents.get("images"):
            with st.expander("View Extracted Images", expanded=False):
                for img in documents["images"]:
                    st.markdown(f"**Image from {img['source']}**")
                    st.image(img["image"])
                    if img.get("text"):
                        st.markdown("**Extracted Text:**")
                        st.text(img["text"])
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
                st.warning("No document content was found to analyze. Please make sure your document contains readable text.")
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
        Error processing the document(s). This could be due to:
        - The document contains no readable text or tables
        - The document format is not supported
        - The document is password protected
        
        Error details: {str(e)}
        
        Please try uploading a different document.
        """)

else:
    # Display instructions when no files are uploaded
    st.info("""
    👋 Welcome to the Document QA Bot!
    
    To get started:
    1. Upload any document using the sidebar
    2. View the extracted data in the expandable section
    3. Ask questions about any content in the document
    
    Example questions:
    - What are the key points mentioned in the document?
    - Can you find any numerical data about [topic]?
    - What does the document say about [specific subject]?
    - Are there any trends or patterns in the data?
    
    Note: Make sure your document contains readable text content.
    """)
