import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os

load_dotenv()
Google_api_key=os.environ.get("GOOGLE_API_KEY")
st.set_page_config(page_title="RAG Chat Bot in GCP", layout="wide")


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=Google_api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGroq(
        groq_api_key=os.environ.get("GROQ_API_KEY"), 
        model_name="llama3-8b-8192"
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=Google_api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever(search_kwargs={"k": 10})
    docs = retriever.invoke(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def main():
    st.header("RAG chatbot")

    st.markdown("""
        <style>
            .chat-message {
                padding: 10px;
                border-radius: 25px;
                margin: 5px;
                position: relative;
                max-width: 80%;
            }
            .user-message {
                background-color: #CCF2F4;
                margin-left: auto;
                text-align: right;
                color: #055052;
                border-top-right-radius: 0;
            }
            .bot-message {
                background-color: #FAD02E;
                margin-right: auto;
                text-align: left;
                color: #665C00;
                border-top-left-radius: 0;
            }
            textarea {
                border: none;
            }
        </style>
    """, unsafe_allow_html=True)


    if 'history' not in st.session_state:
        st.session_state['history'] = []


    user_question = st.text_input("Ask a Question from the PDF Files", key="user_question")

    if st.button("Ask"):
        if user_question:  # Ensure user question is provided
            st.session_state['history'].append(("You", user_question))
            response = user_input(user_question)
            st.session_state['history'].append(("Bot", response))
            st.session_state.user_input = ""


    for role, message in st.session_state['history']:
        if role == "You":
            st.markdown(f"<div class='chat-message user-message'><strong>{role}:</strong> {message}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-message bot-message'><strong>{role}:</strong> {message}</div>", unsafe_allow_html=True)



    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, key="pdf_uploader")
        if st.button("Submit & Process", key="process_button"): 
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
