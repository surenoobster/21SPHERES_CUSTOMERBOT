import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Initialize Streamlit app
st.set_page_config(page_title="Footwear Catalog Chatbot", layout="centered")
st.title("Footwear Catalog Chatbot")
st.write("Ask any question about our footwear catalog!")

# Initialize embeddings
try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
except Exception as e:
    st.error(f"Error initializing HuggingFace embeddings: {e}")
    st.stop()

# Load the FAISS vector store
try:
    vectorstore = FAISS.load_local("footwear_data_huggingface_2.db", embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()
except Exception as e:
    st.error(f"Error loading vectorstore: {e}")
    st.stop()

# Fetch the API key from Streamlit secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]
if not openai_api_key:
    st.error("The OpenAI API key is not set. Please update your Streamlit Secrets.")
    st.stop()

# Load the LLM
try:
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=openai_api_key,
    )
except Exception as e:
    st.error(f"Error initializing the ChatOpenAI model: {e}")
    st.stop()

# Define the prompt template
template = """
You are a chatbot designed to answer questions based solely on the provided footwear catalog information. Your primary goal is to provide accurate, concise, and context-specific answers based on the information in the catalog.

**Instructions:**
1. Use only the content from the provided footwear catalog to generate responses.
2. Do not generate information beyond what is available in the catalog.
3. If you cannot find the answer to a question in the catalog, respond with:
   *"Sorry, I couldn’t find the information you are looking for. Please check the catalog or contact the support team at office@eurosafetyfootwear.com."*

**Footwear Catalog Details**
{context}
**End of Catalog Details**

**Question:** {input}

**Response:**
"""
prompt = ChatPromptTemplate.from_template(template)

doc_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(retriever, doc_chain)

# User input section
user_input = st.text_input("Enter your question about the footwear catalog:", "")

if user_input:
    with st.spinner("Fetching the response..."):
        try:
            response = chain.invoke({"input": user_input})
            st.write("**Response:**", response['answer'])
        except Exception as e:
            st.error(f"Error processing your request: {e}")

# Footer
st.write("---")
st.caption("Powered by 21 Spheres AI")
