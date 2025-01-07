
# Conversational AI with Document Processing

This project integrates conversational AI with document processing, enabling users to interact with PDF and text-based documents via natural language queries. The application uses LangChain for document handling, HuggingFace embeddings for semantic representation, FAISS for fast similarity searches, and ChatGroq for generating relevant responses to user queries.

## Key Features

- **File Upload and Processing**: Supports uploading PDF and text files, with content extracted for further processing.
- **Text Chunking**: Large documents are split into manageable chunks to optimize embedding generation and retrieval.
- **Embedding Generation**: Text chunks are embedded using HuggingFace’s BGE embeddings model and stored in FAISS for fast retrieval.
- **Conversational AI**: Powered by ChatGroq, the AI model generates context-aware responses based on the document content.
- **Session Management**: Users can manage multiple sessions, preserving chat history across interactions.
- **Interactive Chat Interface**: Built with Streamlit, providing an intuitive and responsive interface for real-time querying and conversation.

## Technologies Used

- **LangChain**: For handling document processing, text splitting, and conversational AI chains.
- **HuggingFace**: For generating high-quality embeddings of text data.
- **FAISS**: For fast similarity search and retrieval of document content.
- **ChatGroq**: A conversational AI model for generating answers from the text.
- **Streamlit**: For building the interactive web interface.

## Installation

To run this project locally, follow the steps below:

### 1. Clone the repository

```bash
git clone https://github.com/SHubhamanjk/ChatBot-to-Chat-With-PDF-along-with-Chat-History.git
cd ChatBot-to-Chat-With-PDF-along-with-Chat-History
```

### 2. Set up the virtual environment

Create a virtual environment and activate it:

```bash
python3 -m venv env
source env/bin/activate   # On Windows, use `env\Scripts\activate`
```

### 3. Install dependencies

Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file and add your environment variables (e.g., API keys) if necessary.

### 5. Run the app

Start the Streamlit application:

```bash
streamlit run app.py
```

This will launch the web app in your browser.

## Usage

1. **Upload Files**: Use the sidebar to upload PDF or text files.
2. **Ask Questions**: After the files are processed, you can start asking questions related to the uploaded documents.
3. **Interact**: The application will generate contextually relevant answers based on the document content, and the conversation history will be maintained.

## Example

Once files are uploaded, you can ask questions like:

- "What is the summary of chapter 2?"
- "Can you extract the key points from the introduction?"
- "What are the main findings in the report?"

## Contributing

If you'd like to contribute to this project, feel free to fork the repository, create a branch for your feature or bugfix, and submit a pull request.
