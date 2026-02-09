import streamlit as st
import os 
from dotenv import load_dotenv

# Core LangChain imports
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import RetrievalQA
# Community & Utility imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS # FAISS is more stable for Streamlit local runs

load_dotenv()

st.title("RAG Chatbot")

# Session state for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

@st.cache_resource
def get_vectorstore():
    pdf_path = "./reflexion.pdf"
    if not os.path.exists(pdf_path):
        st.error(f"File {pdf_path} not found!")
        return None
        
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    # Using a proper embedding model (small and fast)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    
    # Creating a FAISS vectorstore in-memory
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

prompt = st.chat_input("Pass your prompt here!")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({'role':'user','content':prompt})
    
    try:
        # Initialize Groq LLM
        # Note: Ensure "openai/gpt-oss-120b" is available in your Groq region
        llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.5)

        # 1. Setup the Prompt Template
        system_prompt = (
            "You are an expert assistant. Use the following context to answer the user query. "
            "If you don't know the answer, say you don't know. "
            "Context: {context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        # 2. Get Vectorstore & Setup Chain
        vectorstore = get_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
        
        # Create the modern RAG chain
        combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

        # 3. Get Response
        with st.spinner("Thinking..."):
            result = rag_chain.invoke({"input": prompt})
            response = result["answer"]
        
        st.chat_message('assistant').markdown(response)
        st.session_state.messages.append({'role':'assistant', 'content':response})

    except Exception as e:
        st.error(f"Error: {str(e)}")