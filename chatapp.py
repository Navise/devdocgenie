import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.title("DevDocGenie React Documentation Assistant")


llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions.
    Provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions:{input}
    """
)

def load_or_create_faiss_index():
    try:
        index_path = "Faiss_index"
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        if os.path.exists(index_path):
            st.write("Loading existing FAISS index")
            vectors = FAISS.load_local(index_path, embeddings=embeddings, allow_dangerous_deserialization=True)
        else:
            st.write("Creating new FAISS index...")
            loader = PyPDFDirectoryLoader("./React")
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            final_documents = text_splitter.split_documents(docs[:20])

            vectors = FAISS.from_documents(final_documents, embeddings)
            vectors.save_local(index_path)

        return vectors
    except Exception as e:
        st.error(f"Error during FAISS index loading or creation: {e}")
        return None

if "vectors" not in st.session_state:
    with st.spinner("Loading or creating FAISS index..."):
        st.session_state.vectors = load_or_create_faiss_index()

prompt1 = st.text_input("Enter Your Question From Document")

if "vectors" in st.session_state and prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    response = retrieval_chain.invoke({'input': prompt1})
    st.write("Chatbot Response:", response['answer'])

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response.get("context", [])):
            st.write(doc.page_content)
            st.write("--------------------------------")
elif prompt1:
    st.warning("Please wait for the FAISS index to load or be created.")
