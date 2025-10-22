import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer
import matplotlib.pyplot as plt

# 🧠 Configuración inicial
st.set_page_config(page_title="Analizador de Frases 💬", page_icon="🧩", layout="centered")
st.markdown("<h1 style='text-align:center; color:#FF4B4B;'>💡 Analizador de Frases en Español</h1>", unsafe_allow_html=True)

# Documentos de ejemplo
default_docs = """El perro ladra fuerte en el parque.
El gato maúlla suavemente durante la noche.
El perro y el gato juegan juntos en el jardín.
Los niños corren y se divierten en el parque.
La música suena muy alta en la fiesta.
Los pájaros cantan hermosas melodías al amanecer."""

# Stemmer en español
stemmer = SnowballStemmer("spanish")

def tokenize_and_stem(text):
    text = text.lower()
    text = re.sub(r'[^a-záéíóúüñ\s]', ' ', text)
    tokens = [t for t in text.split() if len(t) > 1]
    stems = [stemmer.stem(t) for t in tokens]
    return stems

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    text_input = st.text_area("📝 Documentos (uno por línea):", default_docs, height=150)
    question = st.text_input("❓ Escribe tu pregunta:", "¿Dónde juegan el perro y el gato?")

with col2:
    st.markdown("### 💡 Preguntas sugeridas:")
    if st.button("¿Dónde juegan el perro y el gato?", use_container_width=True):
        st.session_state.question = "¿Dónde juegan el perro y el gato?"
        st.rerun()
    if st.button("¿Qué hacen los niños en el parque?", use_container_width=True):
        st.session_state.question = "¿Qué hacen los niños en el parque?"
        st.rerun()
    if st.button("¿Cuándo cantan los pájaros?", use_container_width=True):
        st.session_state.question = "¿Cuándo cantan los pájaros?"
        st.rerun()
    if st.button("¿Dónde suena la música alta?", use_container_width=True):
        st.session_state.question = "¿Dónde suena la música alta?"
        st.rerun()
    if st.button("¿Qué animal maúlla durante la noche?", use_container_width=True):
        st.session_state.question = "¿Qué animal maúlla durante la noche?"
        st.rerun()

if 'question' in st.session_state:
    question = st.session_state.question

# 🔍 Análisis
if st.button("🔎 Analizar", type="primary"):
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]
    
    if len(documents) < 1:
        st.error("⚠️ Ingresa al menos un documento.")
    elif not question.strip():
        st.error("⚠️ Escribe una pregunta.")
    else:
        vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem, min_df=1)
        X = vectorizer.fit_transform(documents)
        df_tfidf = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out(), index=[f"Doc {i+1}" for i in range(len(documents))])
        st.markdown("### 📊 Matriz TF-IDF")
        st.dataframe(df_tfidf.round(3), use_container_width=True)
        
        # Similitud
        question_vec = vectorizer.transform([question])
        similarities = cosine_similarity(question_vec, X).flatten()
        best_idx = similarities.argmax()
        best_doc = documents[best_idx]
        best_score = similarities[best_idx]
        
        # 🎯 Mensajes dinámicos
        st.markdown("### 🎯 Resultado del análisis")
        st.markdown(f"**Tu pregunta:** {question}")
        
        if best_score > 0.4:
            st.success(f"✨ ¡Alta coincidencia! Este documento parece responder muy bien a tu pregunta.\n\n**Respuesta:** {best_doc}")
        elif best_score > 0.2:
            st.warning(f"🤔 Coincidencia media. Podrías reformular la pregunta.\n\n**Respuesta:** {best_doc}")
        else:
            st.error(f"😅 Coincidencia baja. Tal vez prueba con otras palabras.\n\n**Respuesta:** {best_doc}")
        
        st.info(f"📈 Similitud: {best_score:.3f}")
        
        # 🔤 Resaltado de palabras clave coincidentes
        question_tokens = set(tokenize_and_stem(question))
        doc_tokens = set(vectorizer.get_feature_names_out())
        matched_words = question_tokens & doc_tokens
        
        if matched_words:
            st.markdown("### ✨ Palabras clave coincidentes:")
            st.markdown(f"<p style='color:#FF4B4B; font-size:18px;'>{', '.join(sorted(matched_words))}</p>", unsafe_allow_html=True)
        else:
            st.markdown("### 🔍 No se encontraron coincidencias exactas de palabras clave.")
        
              
        # 📈 Visualización de similitudes
        st.markdown("### 📉 Nivel de similitud por documento")
        plt.barh([f"Doc {i+1}" for i in range(len(documents))], similarities, color="#FF4B4B")
        plt.xlabel("Similitud")
        plt.ylabel("Documentos")
        st.pyplot(plt)
