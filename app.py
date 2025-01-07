import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os

load_dotenv()

os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")


def get_pdf_text(uploaded_files):
    """Extract text from multiple uploaded PDF files."""
    text = ""
    for pdf in uploaded_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_file_content(uploaded_files):
    """Extract text content from uploaded text files."""
    text = ""
    for txt_file in uploaded_files:
        text += txt_file.read().decode("utf-8")
    return text

def text_to_chunks(text):
    """Split text into manageable chunks for embeddings."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    return chunks

def get_embeddings(chunks):
    """Generate embeddings from text chunks and store in FAISS."""
    embedding = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embedding)
    return vectorstore

def create_conversation_chain(vectorstore):
    """Set up a conversational chain using ChatGroq and FAISS retriever."""
    llm = ChatGroq(model="gemma2-9b-it",groq_api_key=os.getenv("GROQ_API_KEY"))
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


st.set_page_config(page_title="PDF & Text-based Conversational AI", layout="wide")
st.title("Document based Conversational AI")
st.sidebar.header("Session Management & File Uploads")


if "sessions" not in st.session_state:
    st.session_state.sessions = {1: []}
    st.session_state.current_session = 1
    st.session_state.vectorstores = {}


session_keys = list(st.session_state.sessions.keys())
selected_session = st.sidebar.selectbox("Select Session", session_keys, index=session_keys.index(st.session_state.current_session))

if st.sidebar.button("Start New Session"):
    new_session_key = max(session_keys) + 1
    st.session_state.sessions[new_session_key] = []
    st.session_state.vectorstores[new_session_key] = None
    st.session_state.current_session = new_session_key
    st.session_state.rerun = True


st.session_state.current_session = selected_session
st.session_state.chat_history = st.session_state.sessions[selected_session]


st.sidebar.header("Upload Files")
file_type = st.sidebar.selectbox("File Type", ["PDF", "Text Document"])
uploaded_files = st.sidebar.file_uploader("Upload Files", accept_multiple_files=True, type=["pdf", "txt"] if file_type == "Text Document" else ["pdf"])


if st.sidebar.button("Preprocess Files"):
    if uploaded_files:
        with st.spinner("Processing Files..."):
            if file_type == "PDF":
                text_data = get_pdf_text(uploaded_files)
            else: 
                text_data = get_text_file_content(uploaded_files)

         
            text_chunks = text_to_chunks(text_data)

            
            vectorstore = get_embeddings(text_chunks)
            st.session_state.vectorstores[selected_session] = vectorstore

       
            conversation_chain = create_conversation_chain(vectorstore)
            st.session_state.conversation_chain = conversation_chain

        st.sidebar.success("Files processed successfully! You can now submit questions.")


vectorstore = st.session_state.vectorstores.get(selected_session)


st.subheader("Chat with Your Data")

if uploaded_files and "conversation_chain" in st.session_state:
    user_input = st.text_input("Ask a question:")
    
    if st.button("Submit"):
        if user_input:
            with st.spinner("Fetching answer..."):

                response = st.session_state.conversation_chain.run({"question": user_input, "chat_history": st.session_state.chat_history})
                st.session_state.chat_history.append({"user": user_input, "ai": response})
                st.session_state.sessions[selected_session] = st.session_state.chat_history

    for chat in st.session_state.chat_history:
        if "user" in chat:
            st.markdown(
                f"""
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <img src="https://via.placeholder.com/40/007bff/ffffff?text=U" 
                         style="border-radius: 50%; margin-right: 10px;" />
                    <div style="background-color: black; padding: 10px; border-radius: 10px; flex: 1; text-align: left;">
                        {chat['user']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        if "ai" in chat:
            st.markdown(
                f"""
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <img src="https://via.placeholder.com/40/007bff/ffffff?text=AI" 
                         style="border-radius: 50%; margin-right: 10px;" />
                    <div style="background-color: black; padding: 10px; border-radius: 10px; flex: 1; text-align: left;">
                        {chat['ai']}
                """,
                unsafe_allow_html=True,
            )
else:
    st.info("Please upload files to start interacting.")
