import random

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Disease Risk Prediction", page_icon="🩺", layout="centered")

COLORS = {"Low": "#2E9F6B", "Moderate": "#D9A21B", "High": "#D64545"}

QUOTES = [
    ("Prevention is better than cure.", "Proverb"),
    ("Health is wealth.", "Proverb"),
    ("An apple a day keeps the doctor away.", "Proverb"),
    ("The greatest wealth is health.", "Virgil"),
    ("Early to bed and early to rise makes a man healthy, wealthy, and wise.", "Benjamin Franklin"),
]
# Chest pain shown as a level; each level maps to the dataset's code (1-4) used by the trained models.
CHEST_PAIN = {"None": 4, "Low": 3, "Moderate": 2, "Severe": 1}
CHEST_PAIN_TYPE = {
    "None": "asymptomatic (no chest pain)",
    "Low": "non-anginal pain",
    "Moderate": "atypical angina",
    "Severe": "typical angina",
}
RESULT_QUOTES = {
    "Low": ("Health is wealth.", "Proverb"),
    "Moderate": ("Prevention is better than cure.", "Proverb"),
    "High": ("Prevention is better than cure.", "Proverb"),
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;800&display=swap');
h1, h2, h3, p, label, button, input, textarea, li, .stTabs button {font-family: 'Manrope', system-ui, sans-serif;}
#MainMenu, footer {visibility: hidden;}
.stApp {
  background-image:
    radial-gradient(circle at 8% 0%, rgba(124,58,237,.18), transparent 38%),
    radial-gradient(circle at 96% 8%, rgba(236,72,153,.15), transparent 34%),
    radial-gradient(circle at 50% 100%, rgba(20,184,166,.15), transparent 40%);
}
.block-container {max-width: 780px; padding-top: 1.6rem; padding-bottom: 4rem;}

/* ---------- hero ---------- */
.hero {position: relative; overflow: hidden; border-radius: 26px; padding: 1.9rem 1.8rem 1.5rem; color: #fff;
  background: linear-gradient(120deg, #4F46E5 0%, #7C3AED 45%, #EC4899 100%);
  box-shadow: 0 22px 44px -22px rgba(124,58,237,.8);}
.hero::before {content: ""; position: absolute; inset: 0; pointer-events: none;
  background: radial-gradient(circle at 88% 18%, rgba(255,255,255,.28), transparent 38%),
              radial-gradient(circle at 6% 110%, rgba(255,255,255,.20), transparent 40%);}
.hero > * {position: relative;}
.badge {display: inline-block; padding: .28rem .8rem; border-radius: 99px; font-size: .78rem; font-weight: 800;
  background: rgba(255,255,255,.22); border: 1px solid rgba(255,255,255,.4); margin-bottom: .8rem;}
.hero h1 {color: #fff !important; font-size: 2.3rem; font-weight: 800; letter-spacing: -0.02em; margin: 0 0 .4rem; padding: 0; line-height: 1.12;}
.hero p {color: #fff !important; margin: 0; opacity: .92; max-width: 34rem; line-height: 1.55;}
.ecg {width: 100%; max-width: 440px; height: 44px; margin-top: 1.1rem; display: block;}
.ecg path {fill: none; stroke: #fff; stroke-width: 2.6; stroke-linecap: round; stroke-linejoin: round;
  stroke-dasharray: 900; stroke-dashoffset: 900; animation: draw 2.4s ease-out forwards;}
@keyframes draw {to {stroke-dashoffset: 0;}}
@media (prefers-reduced-motion: reduce) {.ecg path {animation: none; stroke-dashoffset: 0;}}

.chips {display: flex; gap: .5rem; flex-wrap: wrap; margin: 1rem 0 .2rem;}
.chip {padding: .38rem .9rem; border-radius: 99px; font-weight: 800; font-size: .82rem; color: #fff;}
.c1 {background: linear-gradient(90deg, #7C3AED, #0EA5E9);}
.c2 {background: linear-gradient(90deg, #F43F5E, #F97316);}
.c3 {background: linear-gradient(90deg, #14B8A6, #22C55E);}

/* ---------- quotes ---------- */
.quote {margin: 1.1rem 0 .6rem; padding: 1rem 1.2rem 1rem 1.4rem; border-radius: 16px; border-left: 5px solid #7C3AED;
  background: linear-gradient(120deg, rgba(79,70,229,.14), rgba(236,72,153,.14));
  font-style: italic; line-height: 1.5; font-size: 1.04rem;}
.quote small {display: block; font-style: normal; opacity: .7; margin-top: .3rem; font-size: .85rem; font-weight: 600;}

/* ---------- tabs and inputs ---------- */
.stTabs [data-baseweb="tab-list"] {gap: .3rem;}
.stTabs [data-baseweb="tab"] {font-weight: 700; padding: .6rem 1rem; border-radius: 12px 12px 0 0;}
.stTabs [aria-selected="true"] {color: #7C3AED;}
.stTabs [data-baseweb="tab-highlight"] {background-image: linear-gradient(90deg, #7C3AED, #EC4899); background-color: transparent; height: 4px; border-radius: 4px;}
.stTabs [data-baseweb="tab-panel"] {margin-top: .7rem; padding: 1rem 1rem .4rem; border-radius: 18px;
  border: 1px solid rgba(124,58,237,.28); background: rgba(124,58,237,.06);}

.stButton > button {width: 100%; border-radius: 14px; font-weight: 800; padding: .85rem 1rem; font-size: 1.08rem; border: none; transition: transform .15s ease, box-shadow .15s ease;}
.stButton > button[kind="primary"], .stButton button[data-testid="stBaseButton-primary"] {
  background-image: linear-gradient(100deg, #4F46E5, #7C3AED 50%, #EC4899); background-color: #7C3AED; color: #fff;
  box-shadow: 0 14px 28px -14px rgba(124,58,237,.9);}
.stButton > button[kind="primary"]:hover, .stButton button[data-testid="stBaseButton-primary"]:hover {
  transform: translateY(-2px); color: #fff; box-shadow: 0 18px 32px -14px rgba(236,72,153,.9);}

/* ---------- results ---------- */
.section-title {font-size: 1.3rem; font-weight: 800; margin: 2.2rem 0 1rem; letter-spacing: -0.01em;}
.section-title::after {content: ""; display: block; width: 56px; height: 4px; border-radius: 4px; margin-top: .4rem;
  background: linear-gradient(90deg, #7C3AED, #EC4899);}
.rings {display: flex; gap: 1.2rem; justify-content: center; flex-wrap: wrap;}
.ringcard {position: relative; overflow: hidden; flex: 1 1 230px; max-width: 340px; text-align: center; padding: 1.5rem 1rem 1.2rem;
  border: 1px solid rgba(128,128,128,.25); border-radius: 22px;
  background: linear-gradient(160deg, color-mix(in srgb, var(--g1) 14%, transparent), color-mix(in srgb, var(--g2) 8%, transparent));
  box-shadow: 0 16px 34px -22px var(--g1);}
.ringcard::before {content: ""; position: absolute; left: 0; right: 0; top: 0; height: 7px; background: linear-gradient(90deg, var(--g1), var(--g2));}
.ringcard.d {--g1: #7C3AED; --g2: #0EA5E9;}
.ringcard.h {--g1: #F43F5E; --g2: #F97316;}
.ringhead {display: flex; align-items: center; justify-content: center; gap: .55rem; font-weight: 800; font-size: 1.08rem; margin-bottom: 1rem;}
.ico {display: inline-grid; place-items: center; width: 2.1rem; height: 2.1rem; border-radius: 50%; font-size: 1.05rem;
  background: linear-gradient(135deg, var(--g1), var(--g2));}
.ring {position: relative; width: 156px; height: 156px; margin: 0 auto .4rem;}
.ring::before {content: ""; position: absolute; inset: 0; border-radius: 50%;
  background: conic-gradient(var(--c) calc(var(--p) * 1%), rgba(128,128,128,.25) 0);
  -webkit-mask: radial-gradient(farthest-side, transparent calc(100% - 16px), #000 calc(100% - 15px));
  mask: radial-gradient(farthest-side, transparent calc(100% - 16px), #000 calc(100% - 15px));}
.ringtxt {position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center;}
.ringtxt b {font-size: 2.1rem; font-weight: 800; line-height: 1;}
.ringtxt span {font-size: .88rem; font-weight: 800; margin-top: .25rem; color: var(--c);}

.whatif {border: 1px solid rgba(14,165,233,.3); border-radius: 22px; padding: 1.3rem 1.4rem;
  background: linear-gradient(135deg, rgba(14,165,233,.12), rgba(236,72,153,.12));}
.whatif p {margin: 0 0 1rem; line-height: 1.55;}
.barrow {display: grid; grid-template-columns: 9.5rem 1fr 3rem; align-items: center; gap: .7rem; margin: .6rem 0;}
.barl {font-size: .9rem; opacity: .9; font-weight: 600;}
.barv {font-weight: 800; text-align: right;}
.track {height: 14px; border-radius: 99px; background: rgba(128,128,128,.25); overflow: hidden;}
.fill {height: 100%; border-radius: 99px;}
@media (max-width: 520px) {.barrow {grid-template-columns: 1fr 3rem;} .barl {grid-column: 1 / -1;}}

.verdict {margin-top: 1.5rem; padding: 1.1rem 1.3rem; border-radius: 18px; border-left: 7px solid var(--c);
  background: rgba(128,128,128,.12); background: color-mix(in srgb, var(--c) 16%, transparent);
  line-height: 1.55; font-weight: 700; font-size: 1.04rem;}
.note {margin-top: 1.2rem; font-size: .85rem; opacity: .65; line-height: 1.5;}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def load_models():
    return {(d, m): joblib.load(f"models/{d}_{m}.joblib")
            for d in ("diabetes", "heart") for m in ("lr", "rf")}


models = load_models()


def level(p):
    return "Low" if p < 0.33 else "Moderate" if p < 0.66 else "High"


def ring(title, p, kind, icon):
    pct, lvl = round(p * 100), level(p)
    return (f'<div class="ringcard {kind}"><div class="ringhead"><span class="ico">{icon}</span>{title}</div>'
            f'<div class="ring" style="--p:{pct};--c:{COLORS[lvl]}">'
            f'<div class="ringtxt" style="--c:{COLORS[lvl]}"><b>{pct}%</b><span>{lvl} risk</span></div></div></div>')


def quote_html(text, author):
    return f'<div class="quote">&ldquo;{text}&rdquo;<small>{author}</small></div>'


def bar(label, p):
    pct = round(p * 100)
    return (f'<div class="barrow"><span class="barl">{label}</span>'
            f'<div class="track"><div class="fill" style="width:{pct}%;background:{COLORS[level(p)]}"></div></div>'
            f'<span class="barv">{pct}%</span></div>')


st.markdown(
    '<div class="hero"><span class="badge">AI-powered health screening</span>'
    '<h1>Disease Risk Prediction</h1>'
    '<p>Estimate your diabetes and heart disease risk, and see how blood sugar changes the heart estimate.</p>'
    '<svg class="ecg" viewBox="0 0 420 44" preserveAspectRatio="none"><path d="M0 24 H70 L82 24 L92 8 L104 40 '
    'L114 24 H190 L202 24 L212 6 L224 42 L234 24 H310 L322 24 L332 10 L344 38 L354 24 H420"/></svg></div>'
    '<div class="chips"><span class="chip c1">🩸 Diabetes</span><span class="chip c2">❤️ Heart disease</span>'
    '<span class="chip c3">🧠 Machine learning</span></div>',
    unsafe_allow_html=True,
)

if "quote" not in st.session_state:
    st.session_state["quote"] = random.choice(QUOTES)
st.markdown(quote_html(*st.session_state["quote"]), unsafe_allow_html=True)
st.write("")

tab_you, tab_d, tab_h = st.tabs(["About you", "Diabetes details", "Heart details"])

with tab_you:
    c1, c2 = st.columns(2)
    age = c1.number_input("Age", 1, 100, 45)
    sex = c2.selectbox("Sex", ["Male", "Female"])
    glucose = st.number_input("Fasting glucose (mg/dL)", 40, 400, 110,
                              help="Above 120 mg/dL counts as high fasting sugar for the heart model.")
    algo_name = st.radio("Prediction model", ["Random Forest", "Logistic Regression"], horizontal=True)
    algo = "rf" if algo_name == "Random Forest" else "lr"

with tab_d:
    a, b = st.columns(2)
    preg = a.number_input("Pregnancies", 0, 20, 0)
    dbp = b.number_input("Diastolic BP (mmHg)", 30, 140, 75)
    skin = a.number_input("Skin thickness (mm)", 5, 100, 25)
    insulin = b.number_input("Insulin (mu U/ml)", 0, 900, 80)
    bmi = a.number_input("BMI", 10.0, 70.0, 27.0)
    dpf = b.number_input("Diabetes pedigree function", 0.05, 2.5, 0.4)

with tab_h:
    a, b = st.columns(2)
    cp_label = a.selectbox("Chest pain level", list(CHEST_PAIN), index=0)
    cp = CHEST_PAIN[cp_label]
    a.caption(f"Medical type: {CHEST_PAIN_TYPE[cp_label]}")
    sbp = b.number_input("Resting systolic BP (mmHg)", 80, 220, 130)
    chol = a.number_input("Cholesterol (mg/dL)", 100, 600, 220)
    ecg = b.selectbox("Resting ECG (0-2)", [0, 1, 2])
    hr = a.number_input("Max heart rate", 60, 220, 150)
    exang = b.selectbox("Exercise-induced angina", ["No", "Yes"])
    oldpeak = a.number_input("ST depression (oldpeak)", 0.0, 7.0, 1.0)
    slope = b.selectbox("ST slope (1-3)", [1, 2, 3], index=1)
    ca = a.selectbox("Major vessels (0-3)", [0, 1, 2, 3])
    thal = b.selectbox("Thalassemia (3 normal, 6 fixed, 7 reversible)", [3, 6, 7])

st.write("")
if st.button("Check my risk", type="primary"):
    d_row = pd.DataFrame([[preg, glucose, dbp, skin, insulin, bmi, dpf, age]],
                         columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                                  "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"])
    p_d = models[("diabetes", algo)].predict_proba(d_row)[0, 1]

    fbs = int(glucose > 120)
    h_cols = ["age", "sex", "chest_pain_type", "resting_bp", "cholesterol",
              "fasting_blood_sugar", "resting_ecg", "max_heart_rate", "exercise_angina",
              "oldpeak", "slope", "num_major_vessels", "thalassemia"]

    def heart_prob(fbs_value):
        row = pd.DataFrame([[age, int(sex == "Male"), cp, sbp, chol, fbs_value, ecg, hr,
                             int(exang == "Yes"), oldpeak, slope, ca, thal]], columns=h_cols)
        return models[("heart", algo)].predict_proba(row)[0, 1]

    p_h = heart_prob(fbs)
    p_lo, p_hi = heart_prob(0), heart_prob(1)

    st.markdown('<div class="section-title">📊 Your results</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="rings">{ring("Diabetes", p_d, "d", "🩸")}{ring("Heart disease", p_h, "h", "❤️")}</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="section-title">🍬 How blood sugar changes heart risk</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="whatif">'
        f'<p>Same person, only fasting blood sugar changed. Heart risk moves by <b>{p_hi - p_lo:+.0%}</b>.</p>'
        f'{bar("Normal fasting sugar", p_lo)}{bar("High fasting sugar", p_hi)}</div>',
        unsafe_allow_html=True,
    )

    if p_d >= 0.5 and p_h >= 0.5:
        msg, tone = "Risk is elevated for both conditions. Please book a check-up with a doctor soon.", "High"
    elif p_d >= 0.5 or p_h >= 0.5:
        msg, tone = "Risk is elevated for one condition. Consider a medical check-up.", "Moderate"
    else:
        msg, tone = "Risk looks low for both. Keep up regular check-ups and a healthy lifestyle.", "Low"
    st.markdown(f'<div class="verdict" style="--c:{COLORS[tone]}">{msg}</div>', unsafe_allow_html=True)

    st.markdown(quote_html(*RESULT_QUOTES[tone]), unsafe_allow_html=True)

    st.markdown(
        '<div class="note">This is an educational risk estimate, not a medical diagnosis. The diabetes and heart '
        'models are trained on separate datasets, so the blood sugar link is an association the model learned, '
        'not proof of cause.</div>',
        unsafe_allow_html=True,
    )
