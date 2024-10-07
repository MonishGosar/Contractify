import os
from typing import List, Optional

import streamlit as st
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

# Load environment variables
load_dotenv()

@st.cache_data
def get_pdf_text(pdf_doc) -> str:
    """Extract text from a PDF document."""
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

@st.cache_data
def get_text_chunks(text: str) -> List[str]:
    """Split text into chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    return splitter.split_text(text)

@st.cache_resource
def get_vector_store(chunks: List[str]) -> FAISS:
    """Create a vector store from text chunks."""
    embeddings = HuggingFaceEmbeddings()
    return FAISS.from_texts(chunks, embedding=embeddings)

@st.cache_data
def generate_summary(text: str) -> str:
    """Generate a summary of the given text."""
    prompt = f"""
    You are an expert in simplifying complex legal and medical documents. Your task is to summarize the following document in a way that is easily understandable to the average person. Focus on the following:
    Keep it short & crisp
    1. Main purpose of the document (e.g., loan agreement, medical consent form) 3-4 lines
    2. Key terms and conditions in simple language
    3. Important deadlines or dates
    4. Main responsibilities or obligations of each party
    5. Any potential risks or important considerations for the reader

    Present the summary in bullet points, using plain language and avoiding legal jargon. If there are any crucial legal terms, explain them in parentheses.

    Document text:
    {text[:4000]}

    Simplified Summary:
    """
    model = Ollama(model="llama3")
    response = model.predict(prompt)
    return response.strip()

@st.cache_resource
def get_qa_chain():
    """Create a question-answering chain."""
    prompt_template = """
    Answer the question based on the context provided. If the answer is not in the context, say "I don't have enough information to answer that question."
    Context: {context}
    Question: {question}
    Answer:
    """
 
    model = Ollama(model="llama3", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)

def process_document(pdf_doc) -> None:
    """Process the uploaded document and update session state."""
    with st.spinner("Processing document..."):
        st.session_state.raw_text = get_pdf_text(pdf_doc)
        text_chunks = get_text_chunks(st.session_state.raw_text)
        st.session_state.vector_store = get_vector_store(text_chunks)
    st.success("Document processed successfully!")

def generate_document_summary() -> None:
    """Generate and display the document summary."""
    with st.spinner("Generating simplified summary..."):
        st.session_state.summary = generate_summary(st.session_state.raw_text)
    st.success("Summary generated!")

def answer_question(question: str) -> Optional[str]:
    """Answer a question based on the document content."""
    if st.session_state.vector_store is None:
        st.error("Please upload a document first.")
        return None
    
    with st.spinner("Thinking..."):
        docs = st.session_state.vector_store.similarity_search(question)
        chain = get_qa_chain()
        response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    return response["output_text"]

def main():
    st.set_page_config(page_title="Contractual Documentation Summarization", page_icon="📄", layout="wide")
    st.title("Contractual Documentation Summarization & QnA Chatbot📄🤖")
    
    # Initialize session state
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'raw_text' not in st.session_state:
        st.session_state.raw_text = ""
    if 'summary' not in st.session_state:
        st.session_state.summary = ""

    # Sidebar for document upload and summary generation
    with st.sidebar:
        st.header("Document Upload")
        pdf_doc = st.file_uploader("Upload your Contractual Documentation (PDF)", type="pdf")
        
        if pdf_doc:
            process_document(pdf_doc)
            if st.button("Generate Document Summary"):
                generate_document_summary()

    # Main page content
    if st.session_state.summary:
        st.header("Document Summary")
        st.write(st.session_state.summary)
    
    st.header("Ask questions about the document")
    question = st.text_input("Enter your question:")
    
    if question:
        answer = answer_question(question)
        if answer:
            st.subheader("Answer:")
            st.write(answer)

if __name__ == "__main__":
    main()