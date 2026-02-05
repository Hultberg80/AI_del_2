import streamlit as st
import numpy as np
from google import genai
from google.genai import types
import json
import pickle
import os

# Setup 
st.set_page_config(page_title="Handbollschat 1.0", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #ffffff;
    }
    
    /* Chat messages */
    .user-message {
        background-color: #10a37f;
        color: white;
        padding: 12px 16px;
        border-radius: 12px;
        margin: 8px 0;
        width: fit-content;
        max-width: 80%;
        margin-left: auto;
    }
    
    .assistant-message {
        background-color: #ececf1;
        color: #000000;
        padding: 12px 16px;
        border-radius: 12px;
        margin: 8px 0;
        width: fit-content;
        max-width: 80%;
        margin-right: auto;
    }
    
    /* Input area */
    .input-container {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: #ffffff;
        border-top: 1px solid #d1d5db;
        padding: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "embeddings_loaded" not in st.session_state:
    st.session_state.embeddings_loaded = False
    st.session_state.chunks = None
    st.session_state.all_embeddings = None
    st.session_state.client = None

# Setup functions for RAG and embeddings
@st.cache_resource
def initialize_model():
    """Initialize Gemini client"""
    api_key = 'AIzaSyAAcOXlfGSvHI8GlGq9E77J1tz-Y_dLZJU'
    return genai.Client(api_key=api_key)

def create_embeddings(client, text, model="text-embedding-004", task_type="SEMANTIC_SIMILARITY"):
    """Create embeddings for text"""
    return client.models.embed_content(
        model=model, 
        contents=text, 
        config=types.EmbedContentConfig(task_type=task_type)
    )

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def semantic_search(client, query, chunks, embeddings, k=3):
    """Find the k most relevant chunks for a query"""
    query_embedding = create_embeddings(client, query).embeddings[0].values
    similarity_scores = []
    
    for i, chunk_embedding in enumerate(embeddings):
        similarity_score = cosine_similarity(query_embedding, chunk_embedding.values)
        similarity_scores.append((i, similarity_score))
    
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    top_indices = [index for index, _ in similarity_scores[:k]]
    
    return [chunks[index] for index in top_indices]

def generate_response(client, system_prompt, user_message, chunks, all_embeddings):
    """Generate response using RAG"""
    context = "\n".join(semantic_search(client, user_message, chunks, all_embeddings))
    user_prompt = f"Frågan är: {user_message}\n\nKontext från dokumentet:\n{context}"
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=genai.types.GenerateContentConfig(
            system_instruction=system_prompt
        ),
        contents=user_prompt
    )
    return response.text

@st.cache_data
def load_saved_data():
    """Load previously saved chunks and embeddings"""
    chunks_file = "handboll_chunks.json"
    embeddings_file = "handboll_embeddings.pkl"
    
    if os.path.exists(chunks_file) and os.path.exists(embeddings_file):
        # Load chunks from JSON
        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        
        # Load embeddings from pickle
        with open(embeddings_file, "rb") as f:
            all_embeddings = pickle.load(f)
        
        return chunks, all_embeddings, True
    else:
        return None, None, False

# Main UI 
st.title("Handbollschat")
st.markdown("Ställ frågor om handbollsregler")

# Sidebar
with st.sidebar:
    st.header("Inställningar")
    
    # Initialize client
    client = initialize_model()
    
    # Try to load saved data on startup
    chunks, all_embeddings, data_loaded = load_saved_data()
    
    if data_loaded:
        st.session_state.client = client
        st.session_state.chunks = chunks
        st.session_state.all_embeddings = all_embeddings
        st.session_state.embeddings_loaded = True
        st.success("Systemet är redo")
    else:
        st.warning("Ingen data laddad")
        st.info("Kör sparcellen i notebooken först")
    
    st.markdown("---")
    
    if st.button("Rensa chat"):
        st.session_state.chat_history = []
        st.rerun()

# Main chat area
st.markdown("---")

# Display chat history
chat_container = st.container()

with chat_container:
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"""
            <div style='display: flex; justify-content: flex-end; margin: 10px 0;'>
                <div style='background-color: #10a37f; color: white; padding: 12px 16px; border-radius: 12px; max-width: 80%;'>
                    {message['content']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='display: flex; justify-content: flex-start; margin: 10px 0;'>
                <div style='background-color: #ececf1; color: #000000; padding: 12px 16px; border-radius: 12px; max-width: 80%;'>
                    {message['content']}
                </div>
            </div>
            """, unsafe_allow_html=True)

# Input area
st.markdown("---")

col1, col2 = st.columns([1, 0.1])

with col1:
    user_input = st.text_input(
        "Ställ en fråga om handbollsregler...",
        placeholder="T.ex. 'Hur stor ska en handboll vara?'",
        key="user_input"
    )

with col2:
    send_button = st.button("Skicka", use_container_width=True)

# Handle user input
if send_button and user_input:
    if not st.session_state.embeddings_loaded:
        st.error("Ingen data laddad. Kör sparcellen i notebooken först.")
    else:
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Generate response
        with st.spinner("Processar fråga..."):
            try:
                system_prompt = """Du är en hjälpsam assistent som svarar ENDAST baserat på 
                handbollsreglerna som ges i kontexten. Om informationen inte finns i dokumentet, 
                säg tydligt att du inte har denna information. Svara på svenska och förklara 
                enkelt och tydligt."""
                
                response = generate_response(
                    st.session_state.client,
                    system_prompt,
                    user_input,
                    st.session_state.chunks,
                    st.session_state.all_embeddings
                )
                
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response
                })
                
                st.rerun()
            except Exception as e:
                st.error(f"Fel vid generering av svar: {str(e)}")
