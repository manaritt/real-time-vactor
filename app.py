import time
import uuid
from datetime import datetime
import pandas as pd
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range, PointStruct
from fastembed import TextEmbedding

st.set_page_config(page_title="Real-Time Vector Analytics", layout="wide")

COLLECTION_NAME = "live_reviews"


@st.cache_resource
def load_system():
    client = QdrantClient(url="http://localhost:6333")
    model = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return client, model


client, model = load_system()

st.title("📊 Real-Time Vector Analytics using Qdrant")

# --- القائمة الجانبية ---
with st.sidebar:
    st.header("🔍 الفلاتر")
    selected_lang = st.selectbox("اختر اللغة", ["الكل", "ar", "en"])
    time_window = st.selectbox("الوقت", [
        ("آخر 15 دقيقة", 15),
        ("آخر ساعة", 60),
        ("كل الأوقات", None)
    ], format_func=lambda x: x[0])

    st.divider()
    st.header("⚡ أضف تعليق جديد (تجربة حية)")
    new_text = st.text_area("اكتب التعليق هنا:")
    new_lang = st.radio("اللغة", ["ar", "en"], horizontal=True)
    new_cat = st.selectbox("الفئة", ["مدفوعات", "شحن", "منتجات", "دعم"])

    if st.button("إرسال وتحليل فوري 🚀", use_container_width=True):
        if new_text.strip():
            vec = list(model.embed([new_text]))[0].tolist()
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=[PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    payload={
                        "text": new_text,
                        "lang": new_lang,
                        "category": new_cat,
                        "timestamp": int(time.time()),
                        "time_str": datetime.now().strftime("%H:%M:%S")
                    }
                )]
            )
            st.success("تمت الإضافة فوراً!")
            st.rerun()

# تجهيز الفلاتر
conditions = []
if selected_lang != "الكل":
    conditions.append(FieldCondition(key="lang", match=MatchValue(value=selected_lang)))
if time_window[1] is not None:
    cutoff = int(time.time()) - (time_window[1] * 60)
    conditions.append(FieldCondition(key="timestamp", range=Range(gte=cutoff)))

active_filter = Filter(must=conditions) if conditions else None

# --- تقسيم الشاشة ---
col1, col2 = st.columns([1.3, 1])

with col1:
    st.subheader("🧠 البحث بالمعنى")
    query = st.text_input("ابحث بالفكرة (مثلاً: مشاكل بالفيزا):", value="مشكلة في الدفع")

    if query:
        q_vec = list(model.embed([query]))[0].tolist()
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=q_vec,
            query_filter=active_filter,
            limit=5
        ).points

        for r in results:
            score_pct = round(r.score * 100, 1)
            with st.container(border=True):
                st.markdown(f"{r.payload['text']}")
                st.caption(
                    f"نسبة التشابه: {score_pct}% | الفئة: {r.payload['category']} | اللغة: {r.payload['lang']} | الوقت: {r.payload['time_str']}")

with col2:
    recent_records, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=active_filter,
        limit=100,
        with_payload=True,
        with_vectors=True
    )

    st.subheader("🚨 التنبيهات الذكية")
    if recent_records:
        latest = recent_records[0]
        similar = client.query_points(
            collection_name=COLLECTION_NAME,
            query=latest.vector,
            query_filter=active_filter,
            score_threshold=0.60,
            limit=20
        ).points

        if len(similar) >= 3:
            st.error(f"تنبيه! تكررت شكوى مشابهة ({len(similar)}) مرات: \"{latest.payload['text']}\"")
            if len(similar) < 3:
                st.success("الوضع طبيعي، لا توجد مشاكل متكررة.")

            st.subheader("📈 الرسم البياني للتقييمات")
            df = pd.DataFrame([rec.payload for rec in recent_records])
            st.bar_chart(df['category'].value_counts())