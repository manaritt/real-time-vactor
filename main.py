import time
import uuid
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType
from fastembed import TextEmbedding

# الاتصال بالسيرفر المحلي qdrant.exe
client = QdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "live_reviews"

print("جاري تحميل نموذج الذكاء الاصطناعي الخفيف...")
model = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# إنشاء مكان تخزين البيانات
if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    client.create_payload_index(COLLECTION_NAME, "lang", PayloadSchemaType.KEYWORD)
    client.create_payload_index(COLLECTION_NAME, "category", PayloadSchemaType.KEYWORD)
    client.create_payload_index(COLLECTION_NAME, "timestamp", PayloadSchemaType.INTEGER)
    print("✅ تم تجهيز قاعدة البيانات في السيرفر!")

# بيانات المحاكاة
reviews = [
    {"text": "التطبيق يعلق عند الدفع بالبطاقة وينخصم المبلغ مرتين!", "lang": "ar", "category": "مدفوعات"},
    {"text": "The delivery was super fast and packaging is great.", "lang": "en", "category": "شحن"},
    {"text": "فشل في عملية الدفع عبر الفيزا، تظهر رسالة خطأ", "lang": "ar", "category": "مدفوعات"},
    {"text": "المندوب تأخر 3 أيام وما يرد على الهاتف", "lang": "ar", "category": "شحن"},
    {"text": "Payment declined multiple times even though card is active", "lang": "en", "category": "مدفوعات"},
    {"text": "المنتج أصلي وجودته ممتازة جداً أنصح به", "lang": "ar", "category": "منتجات"},
    {"text": "بوابة الدفع لا تعمل أبداً اليوم، نرجو الحل السريع", "lang": "ar", "category": "مدفوعات"},
    {"text": "Customer support solved my issue in 5 minutes!", "lang": "en", "category": "دعم"},
]

print("🚀 بدء إرسال البيانات إلى السيرفر...")
for item in reviews:
    vector = list(model.embed([item["text"]]))[0].tolist()
    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=vector,
        payload={
            "text": item["text"],
            "lang": item["lang"],
            "category": item["category"],
            "timestamp": int(time.time()),
            "time_str": datetime.now().strftime("%H:%M:%S")
        }
    )
    client.upsert(collection_name=COLLECTION_NAME, points=[point])
    print(f"تم إرسال: {item['text']}")
    time.sleep(1)

print("🎉 اكتمل إرسال جميع البيانات بنجاح!")