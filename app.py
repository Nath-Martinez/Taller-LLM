import streamlit as st
import pandas as pd
import plotly.express as px
import tiktoken
from groq import Groq
from sklearn.decomposition import PCA
import numpy as np
import time

# Configuration de la page
st.set_page_config(page_title="Taller LLM - EAFIT", layout="wide")
st.title("🛠️ Desmontando los LLMs")
st.markdown("---")

# --- Sidebar : Configuration ---
with st.sidebar:
    st.header("Configuración")
    groq_api_key = st.text_input("Groq API Key", type="password")
    model_name = st.selectbox("Modelo", ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"])
    
    if not groq_api_key:
        st.warning("⚠️ Ingrese su API Key de Groq para habilitar los Módulos 3 y 4.")

# Onglets pour les modules
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
        encoding = tiktoken.get_encoding("cl100k_base") # Standard OpenAI encoding
        tokens_ids = encoding.encode(text_input)
        tokens_text = [encoding.decode([tid]) for tid in tokens_ids]
        
        # Visualización de tokens con colores alternos
        st.subheader("Visualización de Tokens")
        html_tokens = ""
        for i, token in enumerate(tokens_text):
            color = "#ff4b4b" if i % 2 == 0 else "#1f77b4"
            html_tokens += f'<span style="background-color:{color}; color:white; padding:2px 5px; margin:2px; border-radius:3px;">{token}</span>'
        st.markdown(html_tokens, unsafe_allow_html=True)
        
        # Mapeo y métricas
        st.subheader("Detalles técnicos")
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(pd.DataFrame({"Token": tokens_text, "Token ID": tokens_ids}))
        with col2:
            st.metric("Número de Caracteres", len(text_input))
            st.metric("Número de Tokens", len(tokens_ids))

# --- Módulo 2: Geometría de las Palabras (Embeddings) ---
with tab2:
    st.header("📐 Geometría de las Palabras")
    words_input = st.text_input("Lista de palabras (separadas por coma):", "rey, hombre, mujer, reina, Madrid, España, París, Francia")
    
    if words_input:
        words = [w.strip() for w in words_input.split(",")]
        
        # Simulación de Embeddings (En un entorno real se usaría una API como OpenAI o HuggingFace)
        # Nota: Por simplicidad y portabilidad en el taller, usamos vectores aleatorios fijos por palabra
        np.random.seed(42)
        embedding_dim = 100
        mock_embeddings = np.random.randn(len(words), embedding_dim)
        
        # Aplicar PCA para reducir a 2D
        pca = PCA(n_components=2)
        components = pca.fit_transform(mock_embeddings)
        
        df_pca = pd.DataFrame(components, columns=['x', 'y'])
        df_pca['Palabra'] = words
        
        fig = px.scatter(df_pca, x='x', y='y', text='Palabra', title="Mapa de Embeddings (PCA 2D)")
        fig.update_traces(textposition='top center')
        st.plotly_chart(fig, use_container_width=True)
        st.info("Reto: Verifique si (king) - (man) + (woman) ≈ (queen) en el espacio vectorial.")

# --- Módulo 3 & 4: Inferencia y Métricas ---
with tab3:
    st.header("🤖 Inferencia y Razonamiento")
    sys_prompt = st.text_area("System Prompt:", "Eres un asistente experto en IA.")
    user_prompt = st.text_area("User Prompt:", "¿Qué es el mecanismo de Self-Attention?")
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        temp = st.slider("Temperatura", 0.0, 1.5, 0.7, help="Bajo: Determinsta | Alto: Creativo")
    with col_p2:
        top_p = st.slider("Top-P", 0.0, 1.0, 0.9)

    if st.button("Generar Respuesta"):
        if groq_api_key:
            client = Groq(api_key=groq_api_key)
            
            start_time = time.time()
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temp,
                top_p=top_p
            )
            duration = time.time() - start_time
            
            response_text = completion.choices[0].message.content
            st.write("### Respuesta del Modelo:")
            st.write(response_text)
            
            # --- Módulo 4: Métricas de Desempeño ---
            with tab4:
                st.header("📊 Métricas de Desempeño (Groq)")
                # Groq devuelve metadatos de uso en el objeto completion
                usage = completion.usage
                
                m_col1, m_col2, m_col3 = st.columns(3)
                with m_col1:
                    # Cálculo de Time per Token
                    tpt = (duration / usage.completion_tokens) * 1000 if usage.completion_tokens > 0 else 0
                    st.metric("Time per Token", f"{tpt:.2f} ms")
                with m_col2:
                    # Throughput
                    throughput = usage.completion_tokens / duration if duration > 0 else 0
                    st.metric("Throughput", f"{throughput:.2f} tokens/s")
                with m_col3:
                    st.metric("Total Tokens", usage.total_tokens)
                
                st.write(f"**Tokens Entrada:** {usage.prompt_tokens} | **Tokens Salida:** {usage.completion_tokens}")
        else:
            st.error("Por favor ingrese su API Key de Groq en la barra lateral.")
