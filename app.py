import streamlit as st
import pandas as pd
import plotly.express as px
import tiktoken
from groq import Groq
from sklearn.decomposition import PCA
import numpy as np
import time

# --- Configuration de la page ---
st.set_page_config(page_title="Taller LLM - EAFIT", layout="wide")

# Gestion de l'API Key (via Secrets Streamlit ou saisie manuelle) [cite: 16]
if "GROQ_API_KEY" in st.secrets:
    api_key = st.secrets["GROQ_API_KEY"]
else:
    api_key = st.sidebar.text_input("Introduce tu Groq API Key:", type="password")

# État de la session pour conserver les données entre les onglets 
if 'data' not in st.session_state:
    st.session_state.data = None

st.title("🛠️ Desmontando los LLMs")
st.write("Deep Learning y Arquitecturas Transformer - Prof. Jorge Ivan Padilla Buritica") [cite: 2, 3]
st.markdown("---")

# Création des onglets requis par le taller [cite: 21, 26, 33, 37]
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Tokenizador", 
    "2. Geometría de Embeddings", 
    "3. Inferencia y Groq", 
    "4. Métricas de Rendimiento"
])

# --- Módulo 1: El Laboratorio del Tokenizador ---
with tab1:
    st.header("🧪 El Laboratorio del Tokenizador")
    text_input = st.text_area("Ingrese texto para tokenizar:", "¡Hola EAFIT! Los Transformers son increíbles.")
    
    if text_input:
        encoding = tiktoken.get_encoding("cl100k_base") [cite: 17]
        tokens_ids = encoding.encode(text_input) [cite: 9]
        tokens_text = [encoding.decode([tid]) for tid in tokens_ids]
        
        # Visualisation avec couleurs alternes [cite: 23]
        st.subheader("Visualización de Tokens")
        html_tokens = ""
        for i, token in enumerate(tokens_text):
            color = "#ff4b4b" if i % 2 == 0 else "#1f77b4"
            html_tokens += f'<span style="background-color:{color}; color:white; padding:2px 5px; margin:2px; border-radius:3px; display:inline-block;">{token}</span>'
        st.markdown(html_tokens, unsafe_allow_html=True)
        
        # Métriques et Mapping [cite: 24, 25]
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(pd.DataFrame({"Token": tokens_text, "Token ID": tokens_ids}))
        with col2:
            st.metric("Número de Caracteres", len(text_input))
            st.metric("Número de Tokens", len(tokens_ids))

# --- Módulo 2: Geometría de las Palabras (Embeddings) ---
with tab2:
    st.header("📐 Geometría de las Palabras")
    words_input = st.text_input("Lista de palabras (separadas por coma):", "rey, hombre, mujer, reina, Madrid, España")
    
    if words_input:
        words = [w.strip() for w in words_input.split(",")]
        
        # Simulation d'embeddings (PCA nécessite au moins 2 dimensions) [cite: 29, 30]
        np.random.seed(42)
        mock_embeddings = np.random.randn(len(words), 50)
        
        pca = PCA(n_components=2)
        components = pca.fit_transform(mock_embeddings)
        
        df_pca = pd.DataFrame(components, columns=['x', 'y'])
        df_pca['Palabra'] = words
        
        # Graphique interactif Plotly [cite: 31]
        fig = px.scatter(df_pca, x='x', y='y', text='Palabra', title="Mapa de Embeddings (PCA 2D)")
        fig.update_traces(textposition='top center')
        st.plotly_chart(fig, use_container_width=True)
        st.info("Reto: Verifique visualmente si (king) - (man) + (woman) ≈ (queen)") [cite: 32]

# --- Módulo 3: Inferencia y Razonamiento ---
with tab3:
    st.header("🤖 Inferencia y Razonamiento")
    sys_prompt = st.text_area("System Prompt:", "Eres un asistente experto en IA.") [cite: 36]
    user_prompt = st.text_area("User Prompt:", "¿Qué es el mecanismo de Self-Attention?")
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        temp = st.slider("Temperatura", 0.0, 1.5, 0.7, help="Bajo: Determinista | Alto: Creativo") [cite: 34]
    with col_p2:
        top_p = st.slider("Top-P", 0.0, 1.0, 0.9) [cite: 35]

    if st.button("Generar Respuesta"):
        if not api_key:
            st.error("Por favor, ingrese su API Key.")
        else:
            try:
                client = Groq(api_key=api_key)
                start_time = time.time()
                
                # Utilisation du modèle llama-3.1 car le 3.0 est decommissioned
                completion = client.chat.completions.create(
                    model="llama-3.1-8b-instant", 
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temp,
                    top_p=top_p
                )
                duration = time.time() - start_time
                
                # Stockage des données pour le module 4 
                st.session_state.data = {
                    "response": completion.choices[0].message.content,
                    "usage": completion.usage,
                    "duration": duration
                }
                st.write("### Respuesta del Modelo:")
                st.write(st.session_state.data["response"])
                
            except Exception as e:
                st.error(f"Error: {e}")

# --- Módulo 4: Métricas de Rendimiento ---
with tab4:
    st.header("📊 Métricas de Desempeño (Groq)")
    if st.session_state.data:
        usage = st.session_state.data["usage"]
        duration = st.session_state.data["duration"]
        
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            # Time per Token (ms) [cite: 39]
            tpt = (duration / usage.completion_tokens) * 1000 if usage.completion_tokens > 0 else 0
            st.metric("Time per Token", f"{tpt:.2f} ms")
        with m_col2:
            # Throughput (tokens/s) [cite: 40]
            throughput = usage.completion_tokens / duration if duration > 0 else 0
            st.metric("Throughput", f"{throughput:.2f} t/s")
        with m_col3:
            # Total Tokens [cite: 41]
            st.metric("Total Tokens", usage.total_tokens)
            
        st.write(f"**Tokens Entrada:** {usage.prompt_tokens} | **Tokens Salida:** {usage.completion_tokens}")
    else:
        st.info("Genere una respuesta en el Módulo 3 para ver las métricas.")
