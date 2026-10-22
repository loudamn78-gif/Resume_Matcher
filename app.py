import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# إعداد واجهة التطبيق
st.set_page_config(page_title="AI Resume Matcher", page_icon="📄", layout="centered")

st.title("📄 AI Resume & Job Description Matcher")
st.write("أداة ذكية لمطابقة السيرة الذاتية مع الوصف الوظيفي وحساب نسبة التوافق.")

st.divider()

# 1. رفع ملف السيرة الذاتية
uploaded_file = st.file_uploader("قم برفع السيرة الذاتية (صيغة PDF)", type=["pdf"])

# 2. إدخال الوصف الوظيفي
job_description = st.text_area("أدخل الوصف الوظيفي (Job Description) هنا:", height=150)

# دالة لاستخراج النص من ملف PDF
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# زر التحليل
if st.button("تحليل ومطابقة 🚀"):
    if uploaded_file is not None and job_description.strip() != "":
        with st.spinner("جاري تحليل النص وحساب نسبة المطابقة..."):
            # استخراج النص من السيرة الذاتية
            resume_text = extract_text_from_pdf(uploaded_file)
            
            # تجميع النصوص لمقارنتها
            documents = [resume_text, job_description]
            
            # تحويل النصوص إلى متجهات وحساب التشابه
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(documents)
            
            # حساب Cosine Similarity
            match_percentage = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100
            
            st.success("تم التحليل بنجاح!")
            
            # عرض النتيجة
            st.metric(label="نسبة المطابقة (Matching Score)", value=f"{match_percentage:.1f}%")
            
            # شريط تقدم بصري
            st.progress(int(match_percentage))
            
            # تقييم بسيط للنتيجة
            if match_percentage >= 70:
                st.balloons()
                st.info("💡 **ممتاز!** السيرة الذاتية متوافقة بشكل كبير مع متطلبات الوظيفة.")
            elif match_percentage >= 40:
                st.warning("⚠️ **توافق متوسط:** يُنصح بإضافة بعض المهارات والكلمات المفتاحية المذكورة في الوصف الوظيفي.")
            else:
                st.error("🚨 **توافق ضعيف:** تحتاج السيرة الذاتية إلى تعديل شامل لتتناسب مع هذا الشاغر.")
                
    else:
        st.warning("يرجى رفع ملف السيرة الذاتية وإدخال الوصف الوظيفي أولاً.")