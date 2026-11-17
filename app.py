import random

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Disease Risk Prediction", page_icon="🩺", layout="centered")

COLORS = {"Low": "#16A34A", "Moderate": "#D97706", "High": "#DC2626"}

QUOTES = [
    ("Prevention is better than cure.", "Proverb"),
    ("Health is wealth.", "Proverb"),
    ("An apple a day keeps the doctor away.", "Proverb"),
    ("The greatest wealth is health.", "Virgil"),
    ("Early to bed and early to rise makes a man healthy, wealthy, and wise.", "Benjamin Franklin"),
]
RESULT_QUOTES = {
    "Low": ("Health is wealth.", "Proverb"),
    "Moderate": ("Prevention is better than cure.", "Proverb"),
    "High": ("Prevention is better than cure.", "Proverb"),
}
MEANING = {
    "Low": "Lower likelihood based on the values entered.",
    "Moderate": "Some risk indicators present. Monitor and consider a check-up.",
    "High": "Several risk indicators present. A medical check-up is advised.",
}

# Chest pain shown as a level; each level maps to the dataset's code (1-4) used by the trained models.
CHEST_PAIN = {"None": 4, "Low": 3, "Moderate": 2, "Severe": 1}
CHEST_PAIN_TYPE = {
    "None": "asymptomatic (no chest pain)",
    "Low": "non-anginal pain",
    "Moderate": "atypical angina",
    "Severe": "typical angina",
}
FRIENDLY = {
    "BloodPressure": "Blood pressure", "SkinThickness": "Skin thickness",
    "DiabetesPedigreeFunction": "Family history score", "chest_pain_type": "Chest pain type",
    "resting_bp": "Resting BP", "fasting_blood_sugar": "Fasting blood sugar", "resting_ecg": "Resting ECG",
    "max_heart_rate": "Max heart rate", "exercise_angina": "Exercise angina", "oldpeak": "ST depression",
    "slope": "ST slope", "num_major_vessels": "Major vessels", "thalassemia": "Thalassemia",
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;800&display=swap');
h1, h2, h3, p, label, button, input, textarea, li, .stTabs button {font-family: 'Manrope', system-ui, sans-serif;}
#MainMenu, footer {visibility: hidden;}
.stApp {
  background-image:
    radial-gradient(circle at 0% 0%, rgba(37,99,235,.13), transparent 40%),
    radial-gradient(circle at 100% 100%, rgba(13,148,136,.13), transparent 42%);
}
.block-container {max-width: 800px; padding-top: 1.5rem; padding-bottom: 3rem;}

/* ---------- hero ---------- */
.hero {position: relative; overflow: hidden; border-radius: 22px; padding: 2rem 1.9rem 1.6rem; color: #fff;
  background: linear-gradient(135deg, #0B1F44 0%, #12306B 55%, #1D4ED8 100%);
  box-shadow: 0 24px 48px -26px rgba(29,78,216,.85);}
.hero::before {content: ""; position: absolute; inset: 0; pointer-events: none;
  background-image: linear-gradient(rgba(255,255,255,.06) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.06) 1px, transparent 1px);
  background-size: 30px 30px;
  -webkit-mask-image: linear-gradient(120deg, transparent 20%, #000 100%); mask-image: linear-gradient(120deg, transparent 20%, #000 100%);}
.hero::after {content: ""; position: absolute; right: -60px; top: -60px; width: 220px; height: 220px; border-radius: 50%;
  background: radial-gradient(circle, rgba(94,234,212,.45), transparent 65%); pointer-events: none;}
.hero > * {position: relative; z-index: 1;}
.tag {display: inline-flex; align-items: center; gap: .4rem; padding: .3rem .8rem; border-radius: 99px; font-size: .78rem; font-weight: 700;
  background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.28); margin-bottom: .9rem;}
.tag i {width: 8px; height: 8px; border-radius: 50%; background: #5EEAD4; display: inline-block;}
.hero h1 {color: #fff !important; font-size: 2.35rem; font-weight: 800; letter-spacing: -0.02em; margin: 0 0 .45rem; padding: 0; line-height: 1.1;}
.hero p {color: #fff !important; margin: 0; opacity: .86; max-width: 34rem; line-height: 1.6;}
.ecg {width: 100%; max-width: 440px; height: 44px; margin-top: 1.2rem; display: block;}
.ecg path {fill: none; stroke: #5EEAD4; stroke-width: 2.6; stroke-linecap: round; stroke-linejoin: round;
  stroke-dasharray: 900; stroke-dashoffset: 900; animation: draw 2.4s ease-out forwards;}
@keyframes draw {to {stroke-dashoffset: 0;}}
@media (prefers-reduced-motion: reduce) {.ecg path {animation: none; stroke-dashoffset: 0;}}

.chips {display: flex; gap: .5rem; flex-wrap: wrap; margin: 1rem 0 .1rem;}
.chip {padding: .38rem .9rem; border-radius: 99px; font-weight: 700; font-size: .82rem; color: #fff;}
.c1 {background: linear-gradient(90deg, #1D4ED8, #0D9488);}
.c2 {background: linear-gradient(90deg, #2563EB, #06B6D4);}
.c3 {background: linear-gradient(90deg, #E11D48, #F97316);}

/* ---------- quotes ---------- */
.quote {margin: 1.1rem 0 .5rem; padding: .95rem 1.2rem 1rem 1.4rem; border-radius: 14px; border-left: 5px solid #2563EB;
  background: linear-gradient(120deg, rgba(37,99,235,.12), rgba(13,148,136,.12));
  font-style: italic; line-height: 1.5; font-size: 1.03rem;}
.quote small {display: block; font-style: normal; opacity: .7; margin-top: .3rem; font-size: .85rem; font-weight: 600;}

/* ---------- tabs and inputs ---------- */
.stTabs [data-baseweb="tab-list"] {gap: .3rem;}
.stTabs [data-baseweb="tab"] {font-weight: 700; padding: .6rem 1rem;}
.stTabs [aria-selected="true"] {color: #2563EB;}
.stTabs [data-baseweb="tab-highlight"] {background-image: linear-gradient(90deg, #1D4ED8, #0D9488); background-color: transparent; height: 4px; border-radius: 4px;}
.stTabs [data-baseweb="tab-panel"] {margin-top: .7rem; padding: 1rem 1rem .4rem; border-radius: 16px;
  border: 1px solid rgba(37,99,235,.25); background: rgba(37,99,235,.05);}

.stButton > button {width: 100%; border-radius: 12px; font-weight: 800; padding: .85rem 1rem; font-size: 1.06rem; border: none; transition: transform .15s ease, box-shadow .15s ease;}
.stButton > button[kind="primary"], .stButton button[data-testid="stBaseButton-primary"] {
  background-image: linear-gradient(100deg, #1D4ED8, #0D9488); background-color: #1D4ED8; color: #fff;
  box-shadow: 0 14px 28px -14px rgba(29,78,216,.9);}
.stButton > button[kind="primary"]:hover, .stButton button[data-testid="stBaseButton-primary"]:hover {
  transform: translateY(-2px); color: #fff; box-shadow: 0 18px 32px -14px rgba(13,148,136,.9);}

/* ---------- results ---------- */
.section-title {font-size: 1.25rem; font-weight: 800; margin: 2.2rem 0 1rem; letter-spacing: -0.01em; display: flex; align-items: center; gap: .6rem;}
.section-title::before {content: ""; width: 5px; height: 1.3rem; border-radius: 4px; background: linear-gradient(180deg, #1D4ED8, #0D9488);}
.rgrid {display: grid; grid-template-columns: repeat(auto-fit, minmax(270px, 1fr)); gap: 1rem;}
.rcard {position: relative; overflow: hidden; padding: 1.4rem 1.2rem 1.2rem; border-radius: 20px; border: 1px solid rgba(128,128,128,.25);
  background: linear-gradient(160deg, color-mix(in srgb, var(--g1) 11%, transparent), color-mix(in srgb, var(--g2) 6%, transparent));
  box-shadow: 0 16px 34px -24px var(--g1);}
.rcard::before {content: ""; position: absolute; left: 0; right: 0; top: 0; height: 6px; background: linear-gradient(90deg, var(--g1), var(--g2));}
.rcard.d {--g1: #2563EB; --g2: #06B6D4;}
.rcard.h {--g1: #E11D48; --g2: #F97316;}
.rhead {display: flex; align-items: center; gap: .7rem; margin-bottom: 1rem;}
.ico {display: inline-grid; place-items: center; width: 2.3rem; height: 2.3rem; border-radius: 12px; font-size: 1.1rem;
  background: linear-gradient(135deg, var(--g1), var(--g2));}
.rtitle {font-weight: 800; font-size: 1.08rem; line-height: 1.1;}
.rsub {font-size: .8rem; opacity: .65; margin-top: .15rem;}
.pill {margin-left: auto; padding: .28rem .75rem; border-radius: 99px; font-size: .78rem; font-weight: 800; color: var(--c);
  background: rgba(128,128,128,.15); background: color-mix(in srgb, var(--c) 16%, transparent);
  border: 1px solid color-mix(in srgb, var(--c) 45%, transparent);}
.ring {position: relative; width: 150px; height: 150px; margin: .2rem auto .7rem;}
.ring::before {content: ""; position: absolute; inset: 0; border-radius: 50%;
  background: conic-gradient(var(--c) calc(var(--p) * 1%), rgba(128,128,128,.25) 0);
  -webkit-mask: radial-gradient(farthest-side, transparent calc(100% - 15px), #000 calc(100% - 14px));
  mask: radial-gradient(farthest-side, transparent calc(100% - 15px), #000 calc(100% - 14px));}
.ringtxt {position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center;}
.ringtxt b {font-size: 2.05rem; font-weight: 800; line-height: 1;}
.ringtxt span {font-size: .82rem; font-weight: 700; margin-top: .25rem; opacity: .7;}
.rtext {text-align: center; font-size: .88rem; line-height: 1.45; opacity: .8;}

.panel {border: 1px solid rgba(37,99,235,.28); border-radius: 20px; padding: 1.25rem 1.35rem;
  background: linear-gradient(135deg, rgba(37,99,235,.09), rgba(13,148,136,.09));}
.panel p {margin: 0 0 1rem; line-height: 1.55;}
.barrow {display: grid; grid-template-columns: 9.5rem 1fr 3rem; align-items: center; gap: .7rem; margin: .55rem 0;}
.barl {font-size: .9rem; opacity: .9; font-weight: 600;}
.barv {font-weight: 800; text-align: right; font-size: .92rem;}
.track {height: 12px; border-radius: 99px; background: rgba(128,128,128,.25); overflow: hidden;}
.fill {height: 100%; border-radius: 99px;}
.fgrid {display: grid; grid-template-columns: repeat(auto-fit, minmax(270px, 1fr)); gap: 1rem;}
.fcard {border: 1px solid rgba(128,128,128,.25); border-radius: 18px; padding: 1.1rem 1.2rem; background: rgba(128,128,128,.06);}
.fcard h4 {margin: 0 0 .6rem; font-size: .98rem; font-weight: 800;}
.fcard .fill {background: linear-gradient(90deg, var(--g1), var(--g2));}
.fcard.d {--g1: #2563EB; --g2: #06B6D4;}
.fcard.h {--g1: #E11D48; --g2: #F97316;}
@media (max-width: 520px) {.barrow {grid-template-columns: 1fr 3rem;} .barl {grid-column: 1 / -1;}}

.verdict {margin-top: 1.5rem; padding: 1.1rem 1.3rem; border-radius: 16px; border-left: 7px solid var(--c);
  background: rgba(128,128,128,.12); background: color-mix(in srgb, var(--c) 15%, transparent);
  line-height: 1.55; font-weight: 700; font-size: 1.03rem;}
.note {margin-top: 1.3rem; font-size: .84rem; opacity: .65; line-height: 1.5;}
.foot {margin-top: 2.2rem; padding-top: 1rem; border-top: 1px solid rgba(128,128,128,.25); text-align: center; font-size: .8rem; opacity: .6;}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def load_models():
    return {d: joblib.load(f"models/{d}_rf.joblib") for d in ("diabetes", "heart")}


models = load_models()


def level(p):
    return "Low" if p < 0.33 else "Moderate" if p < 0.66 else "High"


def result_card(title, p, kind, icon):
    pct, lvl = round(p * 100), level(p)
    return (f'<div class="rcard {kind}"><div class="rhead"><span class="ico">{icon}</span>'
            f'<div><div class="rtitle">{title}</div><div class="rsub">Risk estimate</div></div>'
            f'<span class="pill" style="--c:{COLORS[lvl]}">{lvl}</span></div>'
            f'<div class="ring" style="--p:{pct};--c:{COLORS[lvl]}"><div class="ringtxt"><b>{pct}%</b><span>{lvl} risk</span></div></div>'
            f'<div class="rtext">{MEANING[lvl]}</div></div>')


def quote_html(text, author):
    return f'<div class="quote">&ldquo;{text}&rdquo;<small>{author}</small></div>'


def bar(label, p):
    pct = round(p * 100)
    return (f'<div class="barrow"><span class="barl">{label}</span>'
            f'<div class="track"><div class="fill" style="width:{pct}%;background:{COLORS[level(p)]}"></div></div>'
            f'<span class="barv">{pct}%</span></div>')


def factors_card(title, kind, model, top=5):
    imp = pd.Series(model.feature_importances_, index=model.feature_names_in_).sort_values(ascending=False).head(top)
    mx = imp.max() or 1
    rows = "".join(
        f'<div class="barrow"><span class="barl">{FRIENDLY.get(n, n.replace("_", " ").capitalize())}</span>'
        f'<div class="track"><div class="fill" style="width:{v / mx * 100:.0f}%"></div></div>'
        f'<span class="barv">{v * 100:.0f}%</span></div>' for n, v in imp.items())
    return f'<div class="fcard {kind}"><h4>{title}</h4>{rows}</div>'


st.markdown(
    '<div class="hero"><span class="tag"><i></i>AI health risk screening</span>'
    '<h1>Disease Risk Prediction</h1>'
    '<p>Estimate your diabetes and heart disease risk with a Random Forest model, and see how blood sugar changes the heart estimate.</p>'
    '<svg class="ecg" viewBox="0 0 420 44" preserveAspectRatio="none"><path d="M0 24 H70 L82 24 L92 8 L104 40 '
    'L114 24 H190 L202 24 L212 6 L224 42 L234 24 H310 L322 24 L332 10 L344 38 L354 24 H420"/></svg></div>'
    '<div class="chips"><span class="chip c1">🌲 Random Forest</span><span class="chip c2">🩸 Diabetes</span>'
    '<span class="chip c3">❤️ Heart disease</span></div>',
    unsafe_allow_html=True,
)

if "quote" not in st.session_state:
    st.session_state["quote"] = random.choice(QUOTES)
st.markdown(quote_html(*st.session_state["quote"]), unsafe_allow_html=True)
st.write("")

tab_you, tab_d, tab_h = st.tabs(["👤 About you", "🩸 Diabetes details", "❤️ Heart details"])

with tab_you:
    c1, c2 = st.columns(2)
    age = c1.number_input("Age", 1, 100, 45)
    sex = c2.selectbox("Sex", ["Male", "Female"])
    glucose = st.number_input("Fasting glucose (mg/dL)", 40, 400, 110,
                              help="Above 120 mg/dL counts as high fasting sugar for the heart model.")

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
    p_d = models["diabetes"].predict_proba(d_row)[0, 1]

    fbs = int(glucose > 120)
    h_cols = ["age", "sex", "chest_pain_type", "resting_bp", "cholesterol",
              "fasting_blood_sugar", "resting_ecg", "max_heart_rate", "exercise_angina",
              "oldpeak", "slope", "num_major_vessels", "thalassemia"]

    def heart_prob(fbs_value):
        row = pd.DataFrame([[age, int(sex == "Male"), cp, sbp, chol, fbs_value, ecg, hr,
                             int(exang == "Yes"), oldpeak, slope, ca, thal]], columns=h_cols)
        return models["heart"].predict_proba(row)[0, 1]

    p_h = heart_prob(fbs)
    p_lo, p_hi = heart_prob(0), heart_prob(1)

    st.markdown('<div class="section-title">Your results</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="rgrid">{result_card("Diabetes", p_d, "d", "🩸")}'
                f'{result_card("Heart disease", p_h, "h", "❤️")}</div>', unsafe_allow_html=True)

    if p_d >= 0.5 and p_h >= 0.5:
        msg, tone = "Risk is elevated for both conditions. Please book a check-up with a doctor soon.", "High"
    elif p_d >= 0.5 or p_h >= 0.5:
        msg, tone = "Risk is elevated for one condition. Consider a medical check-up.", "Moderate"
    else:
        msg, tone = "Risk looks low for both. Keep up regular check-ups and a healthy lifestyle.", "Low"
    st.markdown(f'<div class="verdict" style="--c:{COLORS[tone]}">{msg}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">How blood sugar changes heart risk</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel">'
        f'<p>Same person, only fasting blood sugar changed. Heart risk moves by <b>{p_hi - p_lo:+.0%}</b>.</p>'
        f'{bar("Normal fasting sugar", p_lo)}{bar("High fasting sugar", p_hi)}</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">What the model weighs most</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="fgrid">{factors_card("Diabetes model", "d", models["diabetes"])}'
                f'{factors_card("Heart model", "h", models["heart"])}</div>', unsafe_allow_html=True)
    st.caption("Importance is measured across the whole training data, not for this one person.")

    st.markdown(quote_html(*RESULT_QUOTES[tone]), unsafe_allow_html=True)
    st.markdown(
        '<div class="note">This is an educational risk estimate, not a medical diagnosis. The diabetes and heart '
        'models are trained on separate datasets, so the blood sugar link is an association the model learned, '
        'not proof of cause.</div>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="foot">Disease Risk Prediction · Random Forest · Streamlit<br>'
            'Artificial Intelligence and Data Science · HNN College of Engineering</div>', unsafe_allow_html=True)
