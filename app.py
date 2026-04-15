import streamlit as st
import pandas as pd
import plotly.express as px
import tiktoken
from groq import Groq
from sklearn.decomposition import PCA
import numpy as np
import time

# --- Configuración de la Página ---
st.set_page_config(page_title="Taller LLM - EAFIT", layout="wide")

# Inicializar estado de sesión para persistencia de datos 
if 'data_inferencia' not in st.session_state:
    st.session_state.data_inferencia = None

# Título y Contexto del Taller [cite: 2, 3, 4]
st.title("🛠️ Desmontando los LLMs")
st.write("Deep Learning y Arquitecturas Transformer - Prof. Jorge Ivan Padilla Buritica")
st.markdown("---")

# --- Sidebar: Configuración ---
with st.sidebar:
    st.header("Configuración")
    # Uso de st.secrets para permanencia o input manual 
    if "GROQ_API_KEY" in st.secrets:
        groq_api_key = st.secrets["GROQ_API_KEY"]
        st.success("API Key cargada desde Secrets")
    else:
        groq_api_key = st.text_input("Introduce tu Groq API Key:", type="password")
    
    # Selección de modelo actualizado (Llama 3.1) 
    model_name = st.selectbox("Modelo", ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"])

# Definición de pestañas [cite: 21, 26, 33, 37]
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Tokenizador", 
    "2. Geometría de Embeddings", 
    "3. Inferencia y Groq", 
    "4. Métricas de Rendimiento"
])

# --- Módulo 1: El Laboratorio del Tokenizador [cite: 21] ---
with tab1:
    st.header("🧪 El Laboratorio del Tokenizador")
    text_input = st.text_area("Ingrese texto para tokenizar:", "¡Hola EAFIT! Los Transformers son increíbles.")
    
    if text_input:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens_ids = encoding.encode(text_input)
        tokens_text = [encoding.decode([tid]) for tid in tokens_ids]
        
        # Visualización de tokens con colores alternos [cite: 23]
        st.subheader("Visualización de Tokens")
        html_tokens = ""
        for i, token in enumerate(tokens_text):
            color = "#ff4b4b" if i % 2 == 0 else "#1f77b4"
            html_tokens += f'<span style="background-color:{color}; color:white; padding:2px 5px; margin:2px; border-radius:3px; display:inline-block;">{token}</span>'
        st.markdown(html_tokens, unsafe_allow_html=True)
        
        # Mapeo y métricas [cite: 24, 25]
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(pd.DataFrame({"Token": tokens_text, "Token ID": tokens_ids}))
        with col2:
            st.metric("Número de Caracteres", len(text_input))
            st.metric("Número de Tokens", len(tokens_ids))

# --- Módulo 2: Geometría de las Palabras [cite: 26] ---
with tab2:
    st.header("📐 Geometría de las Palabras")
    words_input = st.text_input("Lista de palabras (separadas por coma):", "rey, hombre, mujer, reina, Madrid, España")
    
    if words_input:
        words = [w.strip() for w in words_input.split(",")]
        
        # Simulación de Embeddings con PCA [cite: 29, 30]
        np.random.seed(42)
        mock_embeddings = np.random.randn(len(words), 50)
        
        pca = PCA(n_components=2)
        components = pca.fit_transform(mock_embeddings)
        
        df_pca = pd.DataFrame(components, columns=['x', 'y'])
        df_pca['Palabra'] = words
        
        # Gráfica interactiva [cite: 31]
        fig = px.scatter(df_pca, x='x', y='y', text='Palabra', title="Mapa de Embeddings (PCA 2D)")
        fig.update_traces(textposition='top center')
        st.plotly_chart(fig, use_container_width=True)
        st.info("Reto: Verifique si (rey) - (hombre) + (mujer) ≈ (reina) [cite: 32]")

# --- Módulo 3: Inferencia y Razonamiento [cite: 33] ---
with tab3:
    st.header("🤖 Inferencia y Razonamiento")
    sys_prompt = st.text_area("System Prompt:", "Eres un asistente experto en IA.")
    user_prompt = st.text_area("User Prompt:", "¿Qué es el mecanismo de Self-Attention?")
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        temp = st.slider("Temperatura", 0.0, 1.5, 0.7) # [cite: 34]
    with col_p2:
        top_p = st.slider("Top-P", 0.0, 1.0, 0.9) # [cite: 35]

    if st.button("Generar Respuesta"):
        if not groq_api_key:
            st.error("⚠️ Por favor ingrese su API Key de Groq.")
        else:
            try:
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
                
                # Almacenar en session_state para el Módulo 4 [cite: 37]
                st.session_state.data_inferencia = {
                    "respuesta": completion.choices[0].message.content,
                    "usage": completion.usage,
                    "duration": duration
                }
                
                st.write("### Respuesta del Modelo:")
                st.write(st.session_state.data_inferencia["respuesta"])
                
            except Exception as e:
                st.error(f"Error de solicitud: {str(e)}")

# --- Módulo 4: Métricas de Desempeño [cite: 37] ---
with tab4:
    st.header("📊 Métricas de Desempeño (Groq)")
    if st.session_state.data_inferencia:
        data = st.session_state.data_inferencia
        usage = data["usage"]
        duration = data["duration"]
        
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
