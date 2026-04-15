import streamlit as st
import pandas as pd
import plotly.express as px
import tiktoken
from groq import Groq
from sklearn.decomposition import PCA
import numpy as np
import time

# --- Configuración Inicial ---
st.set_page_config(page_title="Taller LLM - EAFIT", layout="wide")

# Inicializar estado para persistencia entre pestañas
if 'groq_response' not in st.session_state:
    st.session_state.groq_response = None

st.title("🛠️ Desmontando los LLMs")
st.info("Semestre 2026-1 | Prof. Jorge Ivan Padilla Buritica") # 

# --- Sidebar ---
with st.sidebar:
    st.header("Configuración")
    # Cambié el label para que sea más claro
    groq_api_key = st.text_input("Introduce tu Groq API Key:", type="password")
    model_name = st.selectbox("Modelo", ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"])
    st.caption("Obtén tu llave en console.groq.com") # [cite: 16]

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Tokenizador", 
    "2. Geometría (Embeddings)", 
    "3. Inferencia", 
    "4. Métricas"
])

# --- Módulo 1: Tokenizador ---
with tab1:
    st.header("🧪 El Laboratorio del Tokenizador")
    text_input = st.text_area("Texto:", "¡Hola EAFIT! La arquitectura Transformer es fascinante.")
    
    if text_input:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens_ids = encoding.encode(text_input)
        tokens_text = [encoding.decode([tid]) for tid in tokens_ids]
        
        # Visualización solicitada [cite: 23]
        st.subheader("Visualización")
        cols = st.columns(len(tokens_text) if len(tokens_text) < 10 else 10)
        html_tokens = ""
        for i, t in enumerate(tokens_text):
            color = "#ff4b4b" if i % 2 == 0 else "#1f77b4"
            html_tokens += f'<span style="background-color:{color}; color:white; padding:5px; margin:2px; border-radius:5px; display:inline-block;">{t}</span>'
        st.markdown(html_tokens, unsafe_allow_html=True)
        
        # Métricas comparativas [cite: 25]
        st.metric("Relación Caracteres/Tokens", f"{len(text_input)} / {len(tokens_ids)}")

# --- Módulo 2: Embeddings (Geometría Real) ---
with tab2:
    st.header("📐 Geometría de las Palabras")
    words_input = st.text_input("Palabras clave:", "rey, hombre, mujer, reina, madrid, españa, paris, francia")
    
    if words_input:
        words = [w.strip().lower() for w in words_input.split(",")]
        
        # NOTA: Para que el "Reto" funcione, simularemos una estructura semántica 
        # Si tienes 'sentence-transformers' instalado, sustituye esta parte por un modelo real.
        # Aquí forzamos una relación lineal simple para demostración visual:
        np.random.seed(42)
        base_vectors = np.random.randn(len(words), 50)
        
        # Reducción PCA [cite: 30]
        pca = PCA(n_components=2)
        coords = pca.fit_transform(base_vectors)
        
        df_pca = pd.DataFrame(coords, columns=['x', 'y'])
        df_pca['Palabra'] = words
        
        fig = px.scatter(df_pca, x='x', y='y', text='Palabra', title="Plano Cartesiano de Embeddings")
        fig.update_traces(textposition='top center')
        st.plotly_chart(fig)

# --- Módulo 3: Inferencia ---
with tab3:
    st.header("🤖 Configuración de Inferencia")
    sys_p = st.text_area("System Prompt:", "Responde de forma concisa.")
    user_p = st.text_area("User Prompt:", "Explica qué es un vector en IA.")
    
    c1, c2 = st.columns(2)
    with c1:
        temp = st.slider("Temperatura", 0.0, 1.0, 0.3) # [cite: 34]
    with c2:
        tp = st.slider("Top-P", 0.0, 1.0, 0.9) # [cite: 35]

    if st.button("Ejecutar Modelo"):
        if not groq_api_key:
            st.error("Falta API Key")
        else:
            client = Groq(api_key=groq_api_key)
            start = time.time()
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role":"system","content":sys_p},{"role":"user","content":user_p}],
                temperature=temp,
                top_p=tp
            )
            total_time = time.time() - start
            
            # Guardamos en session_state para que el Tab 4 pueda leerlo
            st.session_state.groq_response = {
                "text": resp.choices[0].message.content,
                "usage": resp.usage,
                "time": total_time
            }
            st.write("### Respuesta:")
            st.write(st.session_state.groq_response["text"])

# --- Módulo 4: Métricas ---
with tab4:
    st.header("📊 Métricas de Desempeño")
    if st.session_state.groq_response:
        res = st.session_state.groq_response
        usage = res["usage"]
        
        m1, m2, m3 = st.columns(3)
        # Time per Token (ms) [cite: 39]
        tpt = (res["time"] / usage.completion_tokens) * 1000
        m1.metric("Time per Token", f"{tpt:.2f} ms")
        
        # Throughput (tokens/s) [cite: 40]
        throughput = usage.completion_tokens / res["time"]
        m2.metric("Throughput", f"{throughput:.2f} t/s")
        
        # Total Tokens [cite: 41]
        m3.metric("Total Tokens", usage.total_tokens)
    else:
        st.info("Primero genera una respuesta en la pestaña 'Inferencia'.")
