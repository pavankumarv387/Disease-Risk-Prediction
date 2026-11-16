import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Disease Risk Prediction", page_icon="🩺", layout="centered")

ACCENT = "#0F766E"
COLORS = {"Low": "#2E9F6B", "Moderate": "#D9A21B", "High": "#D64545"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;800&display=swap');
h1, h2, h3, p, label, button, input, textarea, li, .stTabs button {
  font-family: 'Manrope', system-ui, sans-serif;
}
#MainMenu, footer {visibility: hidden;}
.block-container {max-width: 760px; padding-top: 2rem; padding-bottom: 4rem;}

.hero {padding: 0.4rem 0 1.4rem;}
.hero h1 {font-size: 2.1rem; font-weight: 800; letter-spacing: -0.02em; margin: 0 0 .35rem; line-height: 1.15;}
.hero p {margin: 0; opacity: .72; max-width: 34rem; line-height: 1.55;}
.ecg {width: 100%%; max-width: 420px; height: 44px; margin-top: 1rem; display: block;}
.ecg path {fill: none; stroke: %(accent)s; stroke-width: 2.4; stroke-linecap: round; stroke-linejoin: round;
  stroke-dasharray: 900; stroke-dashoffset: 900; animation: draw 2.4s ease-out forwards;}
@keyframes draw {to {stroke-dashoffset: 0;}}
@media (prefers-reduced-motion: reduce) {.ecg path {animation: none; stroke-dashoffset: 0;}}

.stTabs [data-baseweb="tab-list"] {gap: .25rem;}
.stTabs [data-baseweb="tab"] {font-weight: 600; padding: .6rem 1rem;}
.stTabs [aria-selected="true"] {color: %(accent)s;}
.stTabs [data-baseweb="tab-highlight"] {background-color: %(accent)s;}

.stButton > button {width: 100%%; border-radius: 12px; font-weight: 800; padding: .75rem 1rem; font-size: 1.05rem; border: none;}
.stButton > button[kind="primary"] {background: %(accent)s; color: #fff;}
.stButton > button[kind="primary"]:hover {background: #0b5f59; color: #fff;}

.section-title {font-size: 1.25rem; font-weight: 800; margin: 2rem 0 .8rem; letter-spacing: -0.01em;}
.rings {display: flex; gap: 1.2rem; justify-content: center; flex-wrap: wrap;}
.ringcard {flex: 1 1 220px; max-width: 320px; text-align: center; padding: 1.3rem 1rem 1.1rem;
  border: 1px solid rgba(128,128,128,.28); border-radius: 18px; background: rgba(128,128,128,.07);}
.ring {position: relative; width: 150px; height: 150px; margin: 0 auto .8rem;}
.ring::before {content: ""; position: absolute; inset: 0; border-radius: 50%%;
  background: conic-gradient(var(--c) calc(var(--p) * 1%%), rgba(128,128,128,.25) 0);
  -webkit-mask: radial-gradient(farthest-side, transparent calc(100%% - 15px), #000 calc(100%% - 14px));
  mask: radial-gradient(farthest-side, transparent calc(100%% - 15px), #000 calc(100%% - 14px));}
.ringtxt {position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center;}
.ringtxt b {font-size: 2rem; font-weight: 800; line-height: 1;}
.ringtxt span {font-size: .85rem; font-weight: 600; margin-top: .25rem; color: var(--c);}
.ringlabel {font-weight: 700; font-size: 1.02rem;}

.whatif {border: 1px solid rgba(128,128,128,.28); border-radius: 18px; padding: 1.2rem 1.3rem; background: rgba(128,128,128,.07);}
.whatif p {margin: 0 0 1rem; line-height: 1.55;}
.barrow {display: grid; grid-template-columns: 9.5rem 1fr 3rem; align-items: center; gap: .7rem; margin: .55rem 0;}
.barl {font-size: .9rem; opacity: .85;}
.barv {font-weight: 800; text-align: right;}
.track {height: 12px; border-radius: 99px; background: rgba(128,128,128,.25); overflow: hidden;}
.fill {height: 100%%; border-radius: 99px;}
@media (max-width: 520px) {.barrow {grid-template-columns: 1fr 3rem;} .barl {grid-column: 1 / -1;}}

.verdict {margin-top: 1.4rem; padding: 1rem 1.2rem; border-radius: 14px; border-left: 6px solid var(--c);
  background: rgba(128,128,128,.09); line-height: 1.55; font-weight: 600;}
.note {margin-top: 1.2rem; font-size: .85rem; opacity: .65; line-height: 1.5;}
</style>
""" % {"accent": ACCENT}

st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def load_models():
    return {(d, m): joblib.load(f"models/{d}_{m}.joblib")
            for d in ("diabetes", "heart") for m in ("lr", "rf")}


models = load_models()


def level(p):
    return "Low" if p < 0.33 else "Moderate" if p < 0.66 else "High"


def ring(title, p):
    pct, lvl = round(p * 100), level(p)
    return (f'<div class="ringcard"><div class="ring" style="--p:{pct};--c:{COLORS[lvl]}">'
            f'<div class="ringtxt" style="--c:{COLORS[lvl]}"><b>{pct}%</b><span>{lvl} risk</span></div></div>'
            f'<div class="ringlabel">{title}</div></div>')


def bar(label, p):
    pct = round(p * 100)
    return (f'<div class="barrow"><span class="barl">{label}</span>'
            f'<div class="track"><div class="fill" style="width:{pct}%;background:{COLORS[level(p)]}"></div></div>'
            f'<span class="barv">{pct}%</span></div>')


st.markdown(
    '<div class="hero"><h1>Disease Risk Prediction</h1>'
    '<p>Estimate your diabetes and heart disease risk, and see how blood sugar changes the heart estimate.</p>'
    '<svg class="ecg" viewBox="0 0 420 44" preserveAspectRatio="none"><path d="M0 24 H70 L82 24 L92 8 L104 40 '
    'L114 24 H190 L202 24 L212 6 L224 42 L234 24 H310 L322 24 L332 10 L344 38 L354 24 H420"/></svg></div>',
    unsafe_allow_html=True,
)

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
    cp = a.selectbox("Chest pain type (1-4)", [1, 2, 3, 4], index=3)
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

    st.markdown('<div class="section-title">Your results</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="rings">{ring("Diabetes", p_d)}{ring("Heart disease", p_h)}</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="section-title">How blood sugar changes heart risk</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="whatif">'
        f'<p>Same person, only fasting blood sugar changed. Heart risk moves by <b>{p_hi - p_lo:+.0%}</b>.</p>'
        f'{bar("Normal fasting sugar", p_lo)}{bar("High fasting sugar", p_hi)}</div>',
        unsafe_allow_html=True,
    )

    top = max(p_d, p_h)
    if p_d >= 0.5 and p_h >= 0.5:
        msg, tone = "Risk is elevated for both conditions. Please book a check-up with a doctor soon.", "High"
    elif p_d >= 0.5 or p_h >= 0.5:
        msg, tone = "Risk is elevated for one condition. Consider a medical check-up.", "Moderate"
    else:
        msg, tone = "Risk looks low for both. Keep up regular check-ups and a healthy lifestyle.", "Low"
    st.markdown(f'<div class="verdict" style="--c:{COLORS[tone]}">{msg}</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="note">This is an educational risk estimate, not a medical diagnosis. The diabetes and heart '
        'models are trained on separate datasets, so the blood sugar link is an association the model learned, '
        'not proof of cause.</div>',
        unsafe_allow_html=True,
    )
