import streamlit as st
import pickle
import re

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ---------------- CLEAN TEXT ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def predict_job(text):
    text = clean_text(text)
    text_vector = vectorizer.transform([text])

    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector)[0]

    return prediction, probability

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Fake Job Detector", page_icon="💼", layout="centered")

# ---------------- DARK THEME CSS ----------------
st.markdown("""
<style>

/* FULL PAGE BACKGROUND */
.stApp {
    background-color: grey;
    color: white;
}

/* APP HEADER BAR */
.header {
    background: linear-gradient(90deg, lightblue, #0072ff);
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    color: white;
    margin-bottom: 20px;
}

/* SUBTITLE */
.subtitle {
    text-align: center;
    color: black;
    margin-bottom: 20px;
}

/* INPUT BOX */
textarea {
    background-color: #1c1f26 !important;
    color: white !important;
    border: 2px solid #00c6ff !important;
    border-radius: 10px !important;
}

/* BUTTON STYLE */
.stButton>button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 10px;
    height: 45px;
    width: 100%;
    font-size: 16px;
}

/* RESULT BOX */
.result {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 22px;
    margin-top: 20px;
}

.real {
    background-color: #1f8b4c;
    color: white;
}

.fake {
    background-color: #c0392b;
    color: white;
}

.card {
    background-color: #1c1f26;
    padding: 15px;
    border-radius: 12px;
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<div class='header'>💼 Fake Job Posting Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-based system to detect fraudulent job postings</div>", unsafe_allow_html=True)

# ---------------- INPUT ----------------
user_input = st.text_area("Enter Job Description:", height=200)

# ---------------- BUTTONS ----------------
col1, col2 = st.columns(2)

predict_btn = col1.button("🔍 Predict")
example_btn = col2.button("💡 Example")

# ---------------- EXAMPLE ----------------
if example_btn:
    user_input = "Earn $5000 weekly from home. No experience required. Pay small fee to start."
    st.text_area("Example Loaded:", value=user_input, height=150)

# ---------------- PREDICTION ----------------
if predict_btn:
    if user_input.strip() == "":
        st.warning("Please enter job description")
    else:
        prediction, probability = predict_job(user_input)

        fake_prob = probability[1] * 100
        real_prob = probability[0] * 100

        if prediction == 1:
            st.markdown(f"<div class='result fake'>❌ FAKE JOB<br>{fake_prob:.2f}% Confidence</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result real'>✅ REAL JOB<br>{real_prob:.2f}% Confidence</div>", unsafe_allow_html=True)

        # STATS CARDS
        st.markdown("<div class='card'>📊 Confidence Breakdown</div>", unsafe_allow_html=True)
        st.progress(int(max(fake_prob, real_prob)))

        st.markdown(f"<div class='card'>Real Job: {real_prob:.2f}%</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='card'>Fake Job: {fake_prob:.2f}%</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("<center style='color:gray'>🚀 Final Year ML Project | Built with Streamlit</center>", unsafe_allow_html=True)