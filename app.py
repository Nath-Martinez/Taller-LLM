import streamlit as st
import pandas as pd
import plotly.express as px
import tiktoken
from groq import Groq
from sklearn.decomposition import PCA
import numpy as np
import time

# --- Configuración de la App ---
st.set_page_config(page_title="Taller LLM - EAFIT", layout="wide")

# Gestión de API Key (Secrets o Manual) [cite: 16]
if "GROQ_API_KEY" in st.secrets:
    api_key = st.secrets["GROQ_API_KEY"]
else:
    api_key = st.sidebar.text_input("Groq API Key:", type="password", help="Pega tu gsk_... aquí")

# Estado para guardar la respuesta y métricas
if 'data' not in st.session_state:
    st.session_state.data = None

st.title("🛠️ Desmontando los LLMs")
st.caption("Curso: Deep Learning y Arquitecturas Transformer - EAFIT 2026-1") # [cite: 2, 4]
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Tokenizador", 
    "2. Embeddings", 
    "3. Inferencia", 
    "4. Métricas"
])

# --- Módulo 1: Tokenizador [cite: 21] ---
with tab1:
    st.header("🧪 El Laboratorio del Tokenizador")
    txt = st.text_area("Ingresa un texto:", "¡Hola EAFIT! La arquitectura Transformer es fascinante.")
    if txt:
        enc = tiktoken.get_encoding("cl100k_base")
        ids = enc.encode(txt)
        toks = [enc.decode([i]) for i in ids]
        
        # Visualización solicitada (colores alternos) [cite: 23, 24]
        html = "".join([f'<span style="background:{("#ff4b4b" if i%2==0 else "#1f77b4")};color:white;padding:3px;margin:2px;border-radius:3px;display:inline-block;">{t}</span>' for i, t in enumerate(toks)])
        st.markdown(html, unsafe_allow_html=True)
        
        # Métrica comparativa [cite: 25]
        st.metric("Comparativa (Caracteres / Tokens)", f"{len(txt)} / {len(ids)}")

# --- Módulo 2: Geometría de los Embeddings [cite: 26] ---
with tab2:
    st.header("📐 Geometría de las Palabras")
    words_input = st.text_input("Lista de palabras (separadas por coma):", "rey, hombre, mujer, reina, madrid, españa")
    if words_input:
        words = [w.strip() for w in words_input.split(",")]
        # Simulación de vectores (PCA requiere al menos 2 componentes) [cite: 30]
        np.random.seed(42)
        mock_vectors = np.random.randn(len(words), 50)
        
        pca = PCA(n_components=2)
        coords = pca.fit_transform(mock_vectors)
        df_geo = pd.DataFrame(coords, columns=['x', 'y'])
        df_geo['Palabra'] = words
        
        # Gráfica interactiva Plotly [cite: 31]
        fig = px.scatter(df_geo, x='x', y='y', text='Palabra', title="Plano Cartesiano de Embeddings")
        fig.update_traces(textposition='top center')
        st.plotly_chart(fig, use_container_width=True)
        st.info("Reto: (king) - (man) + (woman) ≈ (queen)") # [cite: 32]

# --- Módulo 3: Inferencia (Groq API) [cite: 33] ---
with tab3:
    st.header("🤖 Inferencia y Razonamiento")
    sys_p = st.text_area("System Prompt:", "Eres un asistente experto en IA.") # [cite: 36]
    user_p = st.text_area("User Prompt:", "¿Qué es el mecanismo de Self-Attention?")
    
    col1, col2 = st.columns(2)
    with col1:
        temp = st.slider("Temperatura", 0.0, 1.5, 0.7) # [cite: 34]
    with col2:
        tp = st.slider("Top-P", 0.0, 1.0, 0.9) # [cite: 35]

    if st.button("Generar Respuesta"):
        if not api_key:
            st.error("Por favor, ingresa tu API Key en la barra lateral o en Secrets.")
        else:
            try:
                client = Groq(api_key=api_key)
                start_t = time.time()
                
                resp = client.chat.completions.create(
                    model="llama3-8b-8192", # [cite: 47]
                    messages=[
                        {"role": "system", "content": sys_p},
                        {"role": "user", "content": user_p}
                    ],
                    temperature=temp,
                    top_p=tp
                )
                
                end_t = time.time()
                dur = end_t - start_t
                
                # Guardar en sesión para el Módulo 4 
                st.session_state.data = {
                    "text": resp.choices[0].message.content,
                    "usage": resp.usage,
                    "time": dur
                }
                st.subheader("Respuesta del Modelo:")
                st.write(st.session_state.data["text"])
                
            except Exception as e:
                st.error(f"Error de comunicación con Groq: {e}")

# --- Módulo 4: Métricas de Desempeño [cite: 37] ---
with tab4:
    st.header("📊 Métricas de Velocidad")
    if st.session_state.data:
        res = st.session_state.data
        u = res["usage"]
        
        m1, m2, m3 = st.columns(3)
        # Time per Token (ms) [cite: 39]
        tpt = (res["time"] / u.completion_tokens) * 1000
        m1.metric("Time per Token", f"{tpt:.2f} ms")
        
        # Throughput (tokens/s) [cite: 40]
        throughput = u.completion_tokens / res["time"]
        m2.metric("Throughput", f"{throughput:.2f} t/s")
        
        # Total Tokens (Entrada vs Salida) [cite: 41]
        m3.metric("Total Tokens", u.total_tokens)
        st.write(f"Tokens de entrada: **{u.prompt_tokens}** | Tokens de salida: **{u.completion_tokens}**")
    else:
        st.info("Primero genera una respuesta en la pestaña de 'Inferencia'.")
