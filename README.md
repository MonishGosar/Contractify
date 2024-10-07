# Contractual Documentation Summarization & QnA Chatbot

This project provides a Streamlit-based web application for processing, summarizing, and interacting with contractual documents. It uses Retrieval-Augmented Generation (RAG) and other natural language processing techniques to extract information from PDFs, generate summaries, and answer questions based on the document content.

## Features

- PDF document upload and text extraction
- Automatic document summarization
- Question-answering capability using RAG
- User-friendly web interface

## How it Works

1. *Document Processing*: The application extracts text from uploaded PDF documents and splits it into manageable chunks.

2. *Vectorization*: Text chunks are converted into vector embeddings using HuggingFace embeddings and stored in a FAISS vector database.

3. *Summarization*: The application generates a concise summary of the document using the Ollama LLM.

4. *Question Answering*: When a user asks a question, the system:
   - Retrieves relevant text chunks from the vector store (Retrieval)
   - Combines these chunks with the question to prompt the LLM (Augmented Generation)
   - Generates an answer based on the retrieved context and the question

This approach allows the system to provide accurate, context-aware answers even for large documents.


## Requirements

- Python 3.7+
- Streamlit
- Langchain
- Ollama
- PyPDF2
- FAISS
- Hugging Face Transformers

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/MonishGosar/LNRS_HACKATHON_MONISH_GOSAR
   cd LNRS_HACKATHON_MONISH_GOSAR
   ```

2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

3. Install Ollama and download the llama3 model:
   - Follow the installation instructions for Ollama on their [official website](https://ollama.ai/).
   - Once Ollama is installed, run the following command to download the llama3 model:
     ```
     ollama pull llama3
     ```

## Usage

1. Start the Streamlit app:
   ```
   streamlit run ollama.py
   ```

2. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8000`).

3. Use the sidebar to upload a PDF document.

4. Click the "Generate Document Summary" button to create a simplified summary of the document.

5. Use the text input field to ask questions about the document content.

## Ollama and llama3 Model

This project uses Ollama to run the llama3 language model locally. Ollama provides an easy way to run large language models on your local machine. The llama3 model is used for generating document summaries and answering questions.

Make sure you have enough system resources (RAM and CPU) to run the llama3 model efficiently. The performance may vary depending on your hardware specifications.

## Future Scope

- [ ] Implement multi-document analysis and comparison
- [ ] Add support for more document formats (e.g., DOCX, TXT)
- [ ] Enhance the summarization algorithm for better accuracy
- [ ] Implement document version tracking and change detection
- [ ] Add user authentication and document access controls
- [ ] Integrate with popular cloud storage services for easier document management

## Demo

### Video Demonstration
[https://youtu.be/wjh9W7HSqRw]

### Presentation
[https://1drv.ms/p/c/513b0cd02eb821e9/ESLxVwO6wGBChxKrddmahxwBXWIUp0HYsBG7AfnxamIhaA?e=tTc5X0]

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
