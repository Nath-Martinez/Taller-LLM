import streamlit as st
import pandas as pd
import plotly.express as px
import tiktoken
from groq import Groq
from sklearn.decomposition import PCA
import numpy as np
import time

# --- Configuración Permanente ---
st.set_page_config(page_title="Taller LLM - EAFIT", layout="wide")

# Intentar obtener la clave de Secrets o de la barra lateral
if "GROQ_API_KEY" in st.secrets:
    api_key = st.secrets["GROQ_API_KEY"]
else:
    api_key = st.sidebar.text_input("Groq API Key (si no hay secrets):", type="password")

# Estado de la sesión para el Módulo 4
if 'data' not in st.session_state:
    st.session_state.data = None

st.title("🛠️ Desmontando los LLMs")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["1. Tokenizador", "2. Embeddings", "3. Inferencia", "4. Métricas"])

# --- Módulo 1: Tokenizador ---
with tab1:
    st.header("🧪 Tokenización")
    txt = st.text_area("Texto:", "¡Hola EAFIT!")
    if txt:
        enc = tiktoken.get_encoding("cl100k_base")
        ids = enc.encode(txt)
        toks = [enc.decode([i]) for i in ids]
        
        # Visualización de colores [cite: 23]
        html = "".join([f'<span style="background:{("#ff4b4b" if i%2==0 else "#1f77b4")};color:white;padding:3px;margin:2px;border-radius:3px;">{t}</span>' for i, t in enumerate(toks)])
        st.markdown(html, unsafe_allow_html=True)
        st.metric("Comparativa (Chars vs Tokens)", f"{len(txt)} / {len(ids)}") # [cite: 25]

# --- Módulo 2: Embeddings ---
with tab2:
    st.header("📐 Geometría")
    # Generar vectores para el reto: king - man + woman ≈ queen [cite: 32]
    words = st.text_input("Palabras:", "rey, hombre, mujer, reina").split(",")
    if words:
        np.random.seed(42)
        # PCA para reducción a 2D [cite: 30]
        pca = PCA(n_components=2)
        coords = pca.fit_transform(np.random.randn(len(words), 100))
        df = pd.DataFrame(coords, columns=['x', 'y'])
        df['Palabra'] = words
        st.plotly_chart(px.scatter(df, x='x', y='y', text='Palabra')) # [cite: 31]

# --- Módulo 3: Inferencia ---
with tab3:
    st.header("🤖 Inferencia")
    sys = st.text_area("System Prompt:", "Eres un tutor de IA.")
    user = st.text_area("User Prompt:", "¿Qué es el self-attention?")
    
    t = st.slider("Temperatura", 0.0, 1.5, 0.7) # [cite: 34]
    p = st.slider("Top-P", 0.0, 1.0, 0.9) # [cite: 35]

    if st.button("Ejecutar"):
        if not api_key:
            st.error("No hay API Key configurada.")
        elif not user.strip():
            st.warning("Escribe algo en el User Prompt antes de ejecutar.")
        else:
            try:
                client = Groq(api_key=api_key)
                start = time.time()
                # Solicitud a la API [cite: 33]
                resp = client.chat.completions.create(
                    model="llama3-8b-8192", # Asegúrate de que este nombre sea correcto
                    messages=[{"role":"system","content":sys},{"role":"user","content":user}],
                    temperature=t,
                    top_p=p
                )
                dur = time.time() - start
                
                # Guardar métricas [cite: 38]
                st.session_state.data = {"text": resp.choices[0].message.content, "u": resp.usage, "t": dur}
                st.write(st.session_state.data["text"])
            except Exception as e:
                st.error(f"Error de solicitud: {e}")

# --- Módulo 4: Métricas de Desempeño ---
with tab4:
    st.header("📊 Métricas")
    if st.session_state.data:
        d = st.session_state.data
        u = d["u"]
        
        c1, c2, c3 = st.columns(3)
        # Time per Token [cite: 39]
        c1.metric("Time per Token", f"{(d['t']/u.completion_tokens)*1000:.2f} ms")
        # Throughput [cite: 40]
        c2.metric("Throughput", f"{u.completion_tokens/d['t']:.2f} t/s")
        # Total Tokens [cite: 41]
        c3.metric("Total Tokens", u.total_tokens)
    else:
        st.info("Ejecuta el modelo en la pestaña 3 para ver las métricas.")
