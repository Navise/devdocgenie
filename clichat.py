import os
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

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions.
    Please provide the most accurate response based on the question.
    Do not use markdown language, use simple text.
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
            print("Loading existing FAISS index...")
            vectors = FAISS.load_local(index_path, embeddings=embeddings, allow_dangerous_deserialization=True) 
        else:
            print("Creating new FAISS index...")
            loader = PyPDFDirectoryLoader("./React")
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            final_documents = text_splitter.split_documents(docs[:20])

            vectors = FAISS.from_documents(final_documents, embeddings)
            vectors.save_local(index_path)

        return vectors
    except Exception as e:
        print(f"Error during FAISS index loading or creation: {e}")
        return None

print("DevDocGenie React Documentation Assistant")
print("Embedding...")
vectors = load_or_create_faiss_index()

if vectors:
    print("exit or quit to exit the chat!!!")
    print(vectors)

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    while True:
        user_input = input("\nUser: ")

        if user_input.lower() in ["exit", "quit"]:
            break

        response = retrieval_chain.invoke({'input': user_input})

        print("Chatbot:", response['answer'])

else:
    print("Embedding of React Documents failed. Please check the logs and try again.")
