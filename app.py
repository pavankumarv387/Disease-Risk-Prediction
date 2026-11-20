import random

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Disease Risk Prediction", page_icon="🩺", layout="centered")

COLORS = {"Low": "#16A34A", "Moderate": "#D97706", "High": "#DC2626"}
ORDER = {"Low": 0, "Moderate": 1, "High": 2}

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
    "Low": "Few risk signs were found in the details you entered.",
    "Moderate": "Some risk signs were found. Keep an eye on this.",
    "High": "Several risk signs were found. A check-up is advised.",
}
TIPS = {
    "diabetes": {
        "Low": "Keep a balanced diet and stay active.",
        "Moderate": "Cut down sugary drinks, walk daily and recheck your blood sugar in a few months.",
        "High": "Please get a fasting blood sugar or HbA1c test and talk to a doctor.",
    },
    "heart": {
        "Low": "Keep exercising and avoid smoking.",
        "Moderate": "Watch your blood pressure and cholesterol, and add regular exercise.",
        "High": "Please see a doctor and consider an ECG and cholesterol test.",
    },
}
HEADLINE = {
    "Low": ("🟢", "Good news: low risk on both checks", "Keep up your healthy habits and regular check-ups."),
    "Moderate": ("🟠", "Some risk signs were found", "A check-up is a good idea. See the tips below."),
    "High": ("🔴", "Higher risk found: please see a doctor soon", "This tool cannot diagnose. A doctor can run the right tests."),
}

# Words shown to the user -> codes used by the trained models.
CHEST_PAIN = {"None": 4, "Low": 3, "Moderate": 2, "Severe": 1}
CHEST_PAIN_TYPE = {
    "None": "asymptomatic (no chest pain)",
    "Low": "non-anginal pain",
    "Moderate": "atypical angina",
    "Severe": "typical angina",
}
ECG = {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}
SLOPE = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}
THAL = {"Normal": 3, "Fixed defect": 6, "Reversible defect": 7}

FRIENDLY = {
    "BMI": "BMI", "Glucose": "Blood sugar",
    "BloodPressure": "Blood pressure", "SkinThickness": "Skin thickness",
    "DiabetesPedigreeFunction": "Family history score", "chest_pain_type": "Chest pain",
    "resting_bp": "Resting BP", "fasting_blood_sugar": "Fasting blood sugar", "resting_ecg": "Resting ECG",
    "max_heart_rate": "Max heart rate", "exercise_angina": "Exercise chest pain", "oldpeak": "ST depression",
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
.hero p {color: #fff !important; margin: 0; opacity: .88; max-width: 34rem; line-height: 1.6; font-size: 1.04rem;}
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

/* ---------- steps ---------- */
.steps {display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: .8rem; margin: 1.2rem 0 .4rem;}
.step {display: flex; align-items: center; gap: .75rem; padding: .8rem 1rem; border-radius: 16px;
  border: 1px solid rgba(128,128,128,.25); background: rgba(128,128,128,.07);}
.step b {display: inline-grid; place-items: center; flex: none; width: 2rem; height: 2rem; border-radius: 50%; color: #fff; font-size: .95rem;
  background: linear-gradient(135deg, #1D4ED8, #0D9488);}
.step span {font-weight: 700; font-size: .92rem; line-height: 1.3;}

/* ---------- quotes ---------- */
.quote {margin: 1rem 0 .5rem; padding: .9rem 1.2rem .95rem 1.4rem; border-radius: 14px; border-left: 5px solid #2563EB;
  background: linear-gradient(120deg, rgba(37,99,235,.12), rgba(13,148,136,.12));
  font-style: italic; line-height: 1.5; font-size: 1.02rem;}
.quote small {display: block; font-style: normal; opacity: .7; margin-top: .3rem; font-size: .85rem; font-weight: 600;}

/* ---------- tabs and inputs ---------- */
.help {margin: .2rem 0 .8rem; font-size: .9rem; opacity: .78; line-height: 1.5;}
.stTabs [data-baseweb="tab-list"] {gap: .3rem;}
.stTabs [data-baseweb="tab"] {font-weight: 700; padding: .6rem 1rem;}
.stTabs [aria-selected="true"] {color: #2563EB;}
.stTabs [data-baseweb="tab-highlight"] {background-image: linear-gradient(90deg, #1D4ED8, #0D9488); background-color: transparent; height: 4px; border-radius: 4px;}
.stTabs [data-baseweb="tab-panel"] {margin-top: .7rem; padding: 1rem 1rem .4rem; border-radius: 16px;
  border: 1px solid rgba(37,99,235,.25); background: rgba(37,99,235,.05);}
.bmi {display: inline-block; margin: .1rem 0 .8rem; padding: .35rem .8rem; border-radius: 99px; font-weight: 700; font-size: .88rem;
  background: rgba(37,99,235,.14); border: 1px solid rgba(37,99,235,.35);}

.stButton > button, .stDownloadButton > button {width: 100%; border-radius: 12px; font-weight: 800; padding: .85rem 1rem; font-size: 1.06rem; border: none; transition: transform .15s ease, box-shadow .15s ease;}
.stButton > button[kind="primary"], .stButton button[data-testid="stBaseButton-primary"] {
  background-image: linear-gradient(100deg, #1D4ED8, #0D9488); background-color: #1D4ED8; color: #fff;
  box-shadow: 0 14px 28px -14px rgba(29,78,216,.9);}
.stButton > button[kind="primary"]:hover, .stButton button[data-testid="stBaseButton-primary"]:hover {
  transform: translateY(-2px); color: #fff; box-shadow: 0 18px 32px -14px rgba(13,148,136,.9);}

/* ---------- results ---------- */
.section-title {font-size: 1.25rem; font-weight: 800; margin: 2.2rem 0 1rem; letter-spacing: -0.01em; display: flex; align-items: center; gap: .6rem;}
.section-title::before {content: ""; width: 5px; height: 1.3rem; border-radius: 4px; background: linear-gradient(180deg, #1D4ED8, #0D9488);}
.overall {display: flex; align-items: center; gap: 1rem; padding: 1.2rem 1.3rem; border-radius: 20px; border-left: 8px solid var(--c);
  background: rgba(128,128,128,.12); background: color-mix(in srgb, var(--c) 15%, transparent);}
.overall .big {font-size: 2.2rem; line-height: 1;}
.overall h3 {margin: 0; padding: 0; font-size: 1.28rem; font-weight: 800; line-height: 1.25;}
.overall p {margin: .3rem 0 0; opacity: .85; line-height: 1.5;}
.overall small {display: block; margin-top: .45rem; font-weight: 700; opacity: .75;}
.rgrid {display: grid; grid-template-columns: repeat(auto-fit, minmax(270px, 1fr)); gap: 1rem; margin-top: 1rem;}
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
.rtext {text-align: center; font-size: .92rem; line-height: 1.45; opacity: .85; font-weight: 600;}
.tip {margin-top: .9rem; padding: .7rem .9rem; border-radius: 12px; font-size: .88rem; line-height: 1.45;
  background: rgba(128,128,128,.12);}
.tip b {display: block; font-size: .78rem; opacity: .7; margin-bottom: .15rem;}

.panel {border: 1px solid rgba(37,99,235,.28); border-radius: 20px; padding: 1.25rem 1.35rem;
  background: linear-gradient(135deg, rgba(37,99,235,.09), rgba(13,148,136,.09));}
.panel p {margin: 0 0 1rem; line-height: 1.55;}
.barrow {display: grid; grid-template-columns: 9.5rem 1fr 3rem; align-items: center; gap: .7rem; margin: .55rem 0;}
.barl {font-size: .9rem; opacity: .9; font-weight: 600;}
.barv {font-weight: 800; text-align: right; font-size: .92rem;}
.track {height: 12px; border-radius: 99px; background: rgba(128,128,128,.25); overflow: hidden;}
.fill {height: 100%; border-radius: 99px;}
.fgrid {display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem;}
.fcard {border: 1px solid rgba(128,128,128,.25); border-radius: 18px; padding: 1.1rem 1.2rem; background: rgba(128,128,128,.06);}
.fcard h4 {margin: 0 0 .6rem; font-size: .98rem; font-weight: 800;}
.fcard .fill {background: linear-gradient(90deg, var(--g1), var(--g2));}
.fcard.d {--g1: #2563EB; --g2: #06B6D4;}
.fcard.h {--g1: #E11D48; --g2: #F97316;}
@media (max-width: 520px) {.barrow {grid-template-columns: 1fr 3rem;} .barl {grid-column: 1 / -1;}}

.note {margin-top: 1.3rem; font-size: .84rem; opacity: .68; line-height: 1.5;}
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


def result_card(title, p, kind, icon, key):
    pct, lvl = round(p * 100), level(p)
    return (f'<div class="rcard {kind}"><div class="rhead"><span class="ico">{icon}</span>'
            f'<div><div class="rtitle">{title}</div><div class="rsub">Risk score</div></div>'
            f'<span class="pill" style="--c:{COLORS[lvl]}">{lvl}</span></div>'
            f'<div class="ring" style="--p:{pct};--c:{COLORS[lvl]}"><div class="ringtxt"><b>{pct}%</b><span>{lvl} risk</span></div></div>'
            f'<div class="rtext">{MEANING[lvl]}</div>'
            f'<div class="tip"><b>What you can do</b>{TIPS[key][lvl]}</div></div>')


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


def bmi_category(v):
    return "Underweight" if v < 18.5 else "Normal" if v < 25 else "Overweight" if v < 30 else "Obese range"


# ------------------------------------------------------------------ header
st.markdown(
    '<div class="hero"><span class="tag"><i></i>Free health risk check</span>'
    '<h1>Check your health risk in 2 minutes</h1>'
    '<p>Answer a few simple questions to get an easy-to-read estimate of your diabetes and heart disease risk.</p>'
    '<svg class="ecg" viewBox="0 0 420 44" preserveAspectRatio="none"><path d="M0 24 H70 L82 24 L92 8 L104 40 '
    'L114 24 H190 L202 24 L212 6 L224 42 L234 24 H310 L322 24 L332 10 L344 38 L354 24 H420"/></svg></div>'
    '<div class="chips"><span class="chip c1">🔒 Your answers are not saved</span><span class="chip c2">🩸 Diabetes</span>'
    '<span class="chip c3">❤️ Heart disease</span></div>'
    '<div class="steps">'
    '<div class="step"><b>1</b><span>Enter your details</span></div>'
    '<div class="step"><b>2</b><span>Tap Check my risk</span></div>'
    '<div class="step"><b>3</b><span>Read your results</span></div></div>',
    unsafe_allow_html=True,
)

if "quote" not in st.session_state:
    st.session_state["quote"] = random.choice(QUOTES)
st.markdown(quote_html(*st.session_state["quote"]), unsafe_allow_html=True)
st.write("")

tab_basic, tab_lab = st.tabs(["1 · Basic details", "2 · Lab report (optional)"])

# ------------------------------------------------------------------ inputs
with tab_basic:
    st.markdown('<div class="help">Things you can find out at home or from a quick check-up.</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    age = c1.number_input("Age (years)", 1, 100, 40)
    sex = c2.selectbox("Sex", ["Male", "Female"])
    c1, c2 = st.columns(2)
    height = c1.number_input("Height (cm)", 100, 220, 170)
    weight = c2.number_input("Weight (kg)", 25, 250, 70)
    bmi = round(weight / ((height / 100) ** 2), 1)
    st.markdown(f'<div class="bmi">Your BMI: {bmi} · {bmi_category(bmi)}</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    sbp = c1.number_input("Blood pressure: top number (mmHg)", 80, 220, 120,
                          help="Systolic pressure. A typical healthy reading is around 120.")
    dbp = c2.number_input("Blood pressure: bottom number (mmHg)", 30, 140, 80,
                          help="Diastolic pressure. A typical healthy reading is around 80.")
    c1, c2 = st.columns(2)
    glucose = c1.number_input("Fasting blood sugar (mg/dL)", 40, 400, 95,
                              help="Blood sugar after not eating for 8 hours. Above 120 counts as high here.")
    chol = c2.number_input("Cholesterol (mg/dL)", 100, 600, 190,
                           help="Total cholesterol from a blood test.")
    c1, c2 = st.columns(2)
    cp_label = c1.selectbox("Chest pain level", list(CHEST_PAIN), index=0)
    cp = CHEST_PAIN[cp_label]
    c1.caption(f"Medical type: {CHEST_PAIN_TYPE[cp_label]}")
    exang = c2.selectbox("Chest pain during exercise?", ["No", "Yes"])
    if sex == "Female":
        preg = st.number_input("Number of pregnancies", 0, 20, 0)
    else:
        preg = 0

with tab_lab:
    st.markdown('<div class="help">Optional. These come from lab or hospital reports. '
                'If you do not have them, keep the typical values already filled in.</div>', unsafe_allow_html=True)
    a, b = st.columns(2)
    insulin = a.number_input("Insulin level (mu U/ml)", 0, 900, 80)
    skin = b.number_input("Skin fold thickness (mm)", 5, 100, 25)
    dpf = a.number_input("Diabetes family history score", 0.05, 2.5, 0.4,
                         help="A score that reflects diabetes in your relatives. Typical value is about 0.4.")
    hr = b.number_input("Highest heart rate in exercise test", 60, 220, 150)
    ecg_label = a.selectbox("Resting ECG result", list(ECG), index=0)
    slope_label = b.selectbox("Exercise ECG slope", list(SLOPE), index=1)
    oldpeak = a.number_input("ST depression (oldpeak)", 0.0, 7.0, 1.0,
                             help="A measurement from an exercise ECG test.")
    ca = b.selectbox("Blocked vessels seen on scan (0 to 3)", [0, 1, 2, 3])
    thal_label = a.selectbox("Thalassemia blood test", list(THAL), index=0)
    ecg, slope, thal = ECG[ecg_label], SLOPE[slope_label], THAL[thal_label]

st.write("")
go = st.button("Check my risk", type="primary")

# ------------------------------------------------------------------ results
if go:
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
    lvl_d, lvl_h = level(p_d), level(p_h)
    tone = max(lvl_d, lvl_h, key=lambda x: ORDER[x])
    icon, head, sub = HEADLINE[tone]

    st.markdown('<div class="section-title">Your results</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="overall" style="--c:{COLORS[tone]}"><div class="big">{icon}</div><div><h3>{head}</h3>'
        f'<p>{sub}</p><small>Diabetes: {lvl_d} · Heart disease: {lvl_h}</small></div></div>'
        f'<div class="rgrid">{result_card("Diabetes", p_d, "d", "🩸", "diabetes")}'
        f'{result_card("Heart disease", p_h, "h", "❤️", "heart")}</div>',
        unsafe_allow_html=True,
    )

    delta = (p_hi - p_lo) * 100
    if delta > 0.5:
        reading = f"For this profile, high blood sugar raises the heart risk score by about {delta:.0f} points."
    elif delta < -0.5:
        reading = "For this profile, high blood sugar does not raise the heart risk score."
    else:
        reading = "For this profile, blood sugar makes little difference to the heart risk score."
    st.markdown('<div class="section-title">Does blood sugar affect your heart risk?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel">'
        f'<p>We compared the same person with normal vs high fasting blood sugar (above 120 mg/dL). {reading}</p>'
        f'{bar("Normal blood sugar", p_lo)}{bar("High blood sugar", p_hi)}</div>',
        unsafe_allow_html=True,
    )

    with st.expander("Curious how the model decides?"):
        st.markdown(f'<div class="fgrid">{factors_card("Diabetes model", "d", models["diabetes"])}'
                    f'{factors_card("Heart model", "h", models["heart"])}</div>', unsafe_allow_html=True)
        st.caption("These are the inputs the Random Forest model relies on most across all its training data, not for you alone.")

    summary = (
        "DISEASE RISK CHECK SUMMARY\n"
        f"Diabetes risk score: {p_d:.0%} ({lvl_d})\n"
        f"Heart disease risk score: {p_h:.0%} ({lvl_h})\n"
        f"Overall: {head}\n\n"
        f"Age {age}, {sex}, BMI {bmi}, BP {sbp}/{dbp} mmHg, fasting sugar {glucose} mg/dL, cholesterol {chol} mg/dL.\n\n"
        "This is an educational estimate, not a medical diagnosis. Please consult a doctor."
    )
    st.download_button("Download my summary", summary, file_name="risk_summary.txt")

    st.markdown(quote_html(*RESULT_QUOTES[tone]), unsafe_allow_html=True)
    st.markdown(
        '<div class="note">This is an educational risk score from a Random Forest model, not a diagnosis and not the '
        'exact chance of getting a disease. The diabetes and heart models are trained on separate datasets, so the '
        'blood sugar link is an association the model learned, not proof of cause. Please consult a doctor for medical advice.</div>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="foot">Disease Risk Prediction · Random Forest · Streamlit<br>'
            'Artificial Intelligence and Data Science · HNN College of Engineering</div>', unsafe_allow_html=True)
