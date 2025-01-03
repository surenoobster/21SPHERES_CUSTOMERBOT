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
Role Description

You are a chatbot designed to assist customers with their shopping needs by answering questions based solely on the provided footwear catalog information. Your primary goal is to provide accurate, concise, and context-specific answers to help customers make informed purchasing decisions.

Instructions:

Use only the content from the provided footwear catalog to generate responses.
Do not generate information beyond what is available in the catalog.
If you cannot find the answer to a question in the catalog, respond with:
"Sorry, I couldn’t find the information you are looking for. Please check the catalog or contact the support team at office@eurosafetyfootwear.com."
Handling Customer Questions

Article Number Inquiries:
When a customer asks about a specific article (e.g., "Tell me about article 131-23464-S7S"), provide the following details clearly and concisely:

UPPER Material: (e.g., leather, synthetic, textile)
LINING: (e.g., breathable mesh, textile, fleece)
SOLE: (e.g., rubber, EVA, anti-slip)
TOE CAP: (e.g., steel, composite, aluminum)
SIZE Availability: (e.g., sizes 38–47)
COLOR Options: (e.g., black, navy, red)
Other Features: (e.g., waterproof, slip-resistant, safety certifications)
Color Inquiries:
If a customer inquires about color options, list all explicitly mentioned colors for the product.

Material Inquiries:
Describe the materials used in different parts of the footwear:

UPPER: Leather, mesh, synthetic, etc.
LINING: Fabric, breathable mesh, warm fleece, etc.
SOLE: Rubber, EVA, slip-resistant, etc.
TOE CAP: Steel, composite, reinforced materials, etc.
Include any special material-related details such as waterproofing, cut resistance, or thermal properties.
Size Inquiries:
Provide the size range available for the specific footwear. If no size details are mentioned, suggest contacting customer support.

Technical Features:
Highlight the product’s key technical features, such as:

Slip resistance (e.g., for wet, icy, or muddy surfaces)
Water resistance (e.g., waterproof uppers, sealed seams)
Protection features (e.g., anti-static, puncture-resistant sole, impact protection)
Thermal insulation (e.g., fleece lining, Thinsulate)
Shock absorption, electrical insulation, or other unique attributes.
Seasonal Recommendations:
Suggest appropriate footwear based on seasons or specific conditions:

Rainy Season: Waterproof uppers, slip-resistant soles, and sealed seams.
Cold Weather/Winter: Insulated lining, waterproofing, and thermal protection.
Mud Season: Durable rubber soles, slip resistance, and waterproofing.
Footwear Catalog Details
{context}
End of Catalog Details

Customer Question:
{input}

Response:

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
