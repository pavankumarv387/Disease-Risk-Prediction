import random

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="HealthGuard AI | Disease Risk Prediction", page_icon="🩺", layout="wide")

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
    "Moderate": ("🟠", "Some risk signs were found", "A check-up is a good idea. See the recommendations below."),
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

# Typical values used when the user does not know a measurement (clearly flagged in the results).
TYPICAL = {"glucose": 95, "chol": 190, "insulin": 80, "skin": 25, "dpf": 0.4, "hr": 150,
           "ecg": "Normal", "slope": "Flat", "oldpeak": 1.0, "ca": 0, "thal": "Normal"}

# Reference profile used to explain which of YOUR values moved the model score.
REF_D = {"Pregnancies": 0, "Glucose": 95, "BloodPressure": 75, "SkinThickness": 25, "Insulin": 80,
         "BMI": 25.0, "DiabetesPedigreeFunction": 0.4, "Age": 40}
REF_H = {"age": 40, "sex": 1, "chest_pain_type": 4, "resting_bp": 120, "cholesterol": 190,
         "fasting_blood_sugar": 0, "resting_ecg": 0, "max_heart_rate": 150, "exercise_angina": 0,
         "oldpeak": 1.0, "slope": 2, "num_major_vessels": 0, "thalassemia": 3}

FRIENDLY = {
    "BMI": "BMI", "Glucose": "Blood sugar", "BloodPressure": "Blood pressure", "SkinThickness": "Skin thickness",
    "DiabetesPedigreeFunction": "Family history score", "chest_pain_type": "Chest pain",
    "resting_bp": "Resting BP", "fasting_blood_sugar": "Fasting blood sugar", "resting_ecg": "Resting ECG",
    "max_heart_rate": "Max heart rate", "exercise_angina": "Exercise chest pain", "oldpeak": "ST depression",
    "slope": "ST slope", "num_major_vessels": "Major vessels", "thalassemia": "Thalassemia",
}

# General food guidance (common dietary advice for diabetes and heart health, with Indian examples).
FOOD = {
    "diabetes": {
        "title": "Diabetes-friendly eating", "icon": "🩸", "kind": "d",
        "groups": [
            ("good", "Eat more", [
                "Non-starchy vegetables: spinach, beans, gourds, cabbage, cauliflower",
                "Dal, sprouts, chickpeas and other legumes",
                "Whole grains and millets: ragi, jowar, oats, brown rice",
                "Lean protein: fish, eggs, skinless chicken, paneer",
                "Unsweetened curd or buttermilk, and a small handful of nuts",
            ]),
            ("warn", "In moderation", [
                "Rice, chapati and bread: keep portions small",
                "Whole fruits (not juice): one serving at a time",
                "Potato, sweet potato and other starchy vegetables",
                "Honey and jaggery: they are still sugar",
            ]),
            ("bad", "Limit or avoid", [
                "Sugary drinks, packaged juices, sweets and desserts",
                "Maida, white bread and bakery items",
                "Deep-fried snacks and packaged foods",
                "Extra sugar in tea and coffee, and alcohol",
            ]),
        ],
    },
    "heart": {
        "title": "Heart-healthy eating", "icon": "❤️", "kind": "h",
        "groups": [
            ("good", "Eat more", [
                "A variety of vegetables and fruits every day",
                "Whole grains, oats and millets",
                "Dal, beans and other legumes",
                "Oily fish such as sardine or mackerel, about twice a week",
                "Unsalted nuts and seeds such as flaxseed",
            ]),
            ("warn", "In moderation", [
                "Cooking oil: use small amounts of groundnut, sunflower or olive oil",
                "Dairy: choose low-fat milk and curd",
                "Eggs and skinless chicken",
                "Tea and coffee without much sugar",
            ]),
            ("bad", "Limit or avoid", [
                "Salt, pickles, papad and salty snacks (aim for under 5 g of salt a day)",
                "Deep-fried food such as samosa, pakora and chips",
                "Ghee, butter, vanaspati and too much coconut oil",
                "Red and processed meat, sugary drinks and alcohol",
            ]),
        ],
    },
}
GROUP_ICON = {"good": "🟢", "warn": "🟠", "bad": "🔴"}

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
.block-container {max-width: 1120px; padding-top: 1.2rem; padding-bottom: 3rem;}

/* ---------- brand bar ---------- */
.brand {display: flex; align-items: center; gap: .8rem; margin-bottom: 1rem;}
.logo {display: inline-grid; place-items: center; width: 2.6rem; height: 2.6rem; border-radius: 14px; font-size: 1.3rem;
  background: linear-gradient(135deg, #1D4ED8, #0D9488);}
.brand b {display: block; font-size: 1.2rem; font-weight: 800; letter-spacing: -0.01em; line-height: 1.1;}
.brand small {opacity: .7; font-weight: 600;}
.brand .bt {margin-left: auto; padding: .3rem .8rem; border-radius: 99px; font-size: .78rem; font-weight: 700;
  border: 1px solid rgba(13,148,136,.5); background: rgba(13,148,136,.14);}

/* ---------- hero ---------- */
.hero {position: relative; overflow: hidden; border-radius: 24px; padding: 2.3rem 2.2rem 1.8rem; color: #fff;
  background: linear-gradient(135deg, #0B1F44 0%, #12306B 55%, #1D4ED8 100%);
  box-shadow: 0 24px 48px -26px rgba(29,78,216,.85);}
.hero::before {content: ""; position: absolute; inset: 0; pointer-events: none;
  background-image: linear-gradient(rgba(255,255,255,.06) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.06) 1px, transparent 1px);
  background-size: 30px 30px;
  -webkit-mask-image: linear-gradient(120deg, transparent 20%, #000 100%); mask-image: linear-gradient(120deg, transparent 20%, #000 100%);}
.hero::after {content: ""; position: absolute; right: -60px; top: -60px; width: 260px; height: 260px; border-radius: 50%;
  background: radial-gradient(circle, rgba(94,234,212,.45), transparent 65%); pointer-events: none;}
.hero > * {position: relative; z-index: 1;}
.tag {display: inline-flex; align-items: center; gap: .4rem; padding: .3rem .8rem; border-radius: 99px; font-size: .78rem; font-weight: 700;
  background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.28); margin-bottom: .9rem;}
.tag i {width: 8px; height: 8px; border-radius: 50%; background: #5EEAD4; display: inline-block;}
.hero h1 {color: #fff !important; font-size: 2.6rem; font-weight: 800; letter-spacing: -0.02em; margin: 0 0 .5rem; padding: 0; line-height: 1.1;}
.hero p {color: #fff !important; margin: 0; opacity: .88; max-width: 36rem; line-height: 1.6; font-size: 1.05rem;}
.ecg {width: 100%; max-width: 460px; height: 44px; margin-top: 1.2rem; display: block;}
.ecg path {fill: none; stroke: #5EEAD4; stroke-width: 2.6; stroke-linecap: round; stroke-linejoin: round;
  stroke-dasharray: 900; stroke-dashoffset: 900; animation: draw 2.4s ease-out forwards;}
@keyframes draw {to {stroke-dashoffset: 0;}}
@media (prefers-reduced-motion: reduce) {.ecg path {animation: none; stroke-dashoffset: 0;}}

.chips {display: flex; gap: .5rem; flex-wrap: wrap; margin: 1rem 0 .1rem;}
.chip {padding: .4rem .95rem; border-radius: 99px; font-weight: 700; font-size: .82rem; color: #fff;}
.c1 {background: linear-gradient(90deg, #2563EB, #06B6D4);}
.c2 {background: linear-gradient(90deg, #E11D48, #F97316);}
.c3 {background: rgba(128,128,128,.35);}

/* ---------- quotes ---------- */
.quote {margin: 1rem 0 .5rem; padding: .9rem 1.2rem .95rem 1.4rem; border-radius: 14px; border-left: 5px solid #2563EB;
  background: linear-gradient(120deg, rgba(37,99,235,.12), rgba(13,148,136,.12));
  font-style: italic; line-height: 1.5; font-size: 1.02rem;}
.quote small {display: block; font-style: normal; opacity: .7; margin-top: .3rem; font-size: .85rem; font-weight: 600;}

/* ---------- step navigation (tabs) and inputs ---------- */
.help {margin: .2rem 0 .8rem; font-size: .9rem; opacity: .78; line-height: 1.5;}
.sec {display: flex; align-items: center; gap: .5rem; margin: 1.1rem 0 .5rem; font-weight: 800; font-size: 1.05rem;}
.sec::after {content: ""; flex: 1; height: 1px; background: rgba(128,128,128,.3);}
.notice {margin: .2rem 0 1rem; padding: .85rem 1rem; border-radius: 14px; font-size: .9rem; line-height: 1.5;
  border: 1px solid rgba(37,99,235,.35); background: rgba(37,99,235,.10);}
.notice b {display: block; margin-bottom: .15rem;}
.stTabs [data-baseweb="tab-list"] {gap: .35rem; padding: .35rem; border-radius: 16px; background: rgba(128,128,128,.12);}
.stTabs [data-baseweb="tab"] {font-weight: 700; padding: .6rem 1.2rem; border-radius: 12px; height: auto;}
.stTabs [aria-selected="true"] {background-image: linear-gradient(100deg, #1D4ED8, #0D9488); color: #fff;}
.stTabs [aria-selected="true"] p {color: #fff;}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {display: none;}
.stTabs [data-baseweb="tab-panel"] {margin-top: .8rem; padding: 1rem 1.2rem .6rem; border-radius: 18px;
  border: 1px solid rgba(37,99,235,.25); background: rgba(37,99,235,.05);}
.bmi {display: inline-block; margin: .1rem 0 .4rem; padding: .35rem .8rem; border-radius: 99px; font-weight: 700; font-size: .88rem;
  background: rgba(37,99,235,.14); border: 1px solid rgba(37,99,235,.35);}

.stButton > button, .stDownloadButton > button {width: 100%; border-radius: 12px; font-weight: 800; padding: .85rem 1rem; font-size: 1.06rem; border: none; transition: transform .15s ease, box-shadow .15s ease;}
.stButton > button[kind="primary"], .stButton button[data-testid="stBaseButton-primary"] {
  background-image: linear-gradient(100deg, #1D4ED8, #0D9488); background-color: #1D4ED8; color: #fff;
  box-shadow: 0 14px 28px -14px rgba(29,78,216,.9);}
.stButton > button[kind="primary"]:hover, .stButton button[data-testid="stBaseButton-primary"]:hover {
  transform: translateY(-2px); color: #fff; box-shadow: 0 18px 32px -14px rgba(13,148,136,.9);}

/* ---------- results dashboard ---------- */
.section-title {font-size: 1.3rem; font-weight: 800; margin: 2.4rem 0 .3rem; letter-spacing: -0.01em; display: flex; align-items: center; gap: .6rem;}
.section-title::before {content: ""; width: 5px; height: 1.3rem; border-radius: 4px; background: linear-gradient(180deg, #1D4ED8, #0D9488);}
.section-sub {margin: 0 0 1rem; opacity: .72; font-size: .92rem; line-height: 1.5;}
.overall {display: flex; align-items: center; gap: 1rem; padding: 1.3rem 1.4rem; border-radius: 20px; border-left: 8px solid var(--c);
  background: rgba(128,128,128,.12); background: color-mix(in srgb, var(--c) 15%, transparent);}
.overall .big {font-size: 2.3rem; line-height: 1;}
.overall h3 {margin: 0; padding: 0; font-size: 1.35rem; font-weight: 800; line-height: 1.25;}
.overall p {margin: .3rem 0 0; opacity: .85; line-height: 1.5;}
.overall small {display: block; margin-top: .45rem; font-weight: 700; opacity: .75;}
.assume {margin-top: .8rem; padding: .7rem 1rem; border-radius: 12px; font-size: .88rem; line-height: 1.5;
  border: 1px dashed rgba(217,119,6,.6); background: rgba(217,119,6,.10);}
.rgrid {display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem;}
.rcard {position: relative; overflow: hidden; padding: 1.5rem 1.3rem 1.3rem; border-radius: 22px; border: 1px solid rgba(128,128,128,.25);
  background: linear-gradient(160deg, color-mix(in srgb, var(--g1) 11%, transparent), color-mix(in srgb, var(--g2) 6%, transparent));
  box-shadow: 0 16px 34px -24px var(--g1);}
.rcard::before {content: ""; position: absolute; left: 0; right: 0; top: 0; height: 6px; background: linear-gradient(90deg, var(--g1), var(--g2));}
.rcard.d, .food.d, .fcard.d {--g1: #2563EB; --g2: #06B6D4;}
.rcard.h, .food.h, .fcard.h {--g1: #E11D48; --g2: #F97316;}
.rhead {display: flex; align-items: center; gap: .7rem; margin-bottom: 1rem;}
.ico {display: inline-grid; place-items: center; width: 2.3rem; height: 2.3rem; border-radius: 12px; font-size: 1.1rem;
  background: linear-gradient(135deg, var(--g1), var(--g2));}
.rtitle {font-weight: 800; font-size: 1.08rem; line-height: 1.1;}
.rsub {font-size: .8rem; opacity: .65; margin-top: .15rem;}
.pill {margin-left: auto; padding: .28rem .75rem; border-radius: 99px; font-size: .78rem; font-weight: 800; color: var(--c);
  background: rgba(128,128,128,.15); background: color-mix(in srgb, var(--c) 16%, transparent);
  border: 1px solid color-mix(in srgb, var(--c) 45%, transparent);}
.ring {position: relative; width: 156px; height: 156px; margin: .2rem auto .7rem;}
.ring::before {content: ""; position: absolute; inset: 0; border-radius: 50%;
  background: conic-gradient(var(--c) calc(var(--p) * 1%), rgba(128,128,128,.25) 0);
  -webkit-mask: radial-gradient(farthest-side, transparent calc(100% - 15px), #000 calc(100% - 14px));
  mask: radial-gradient(farthest-side, transparent calc(100% - 15px), #000 calc(100% - 14px));}
.ringtxt {position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center;}
.ringtxt b {font-size: 2.2rem; font-weight: 800; line-height: 1;}
.ringtxt span {font-size: .78rem; font-weight: 700; margin-top: .25rem; opacity: .7;}
.lvl {text-align: center; font-size: 1.5rem; font-weight: 800; color: var(--c); margin-bottom: .2rem;}
.rtext {text-align: center; font-size: .92rem; line-height: 1.45; opacity: .85; font-weight: 600;}

.panel {border: 1px solid rgba(37,99,235,.28); border-radius: 20px; padding: 1.25rem 1.35rem;
  background: linear-gradient(135deg, rgba(37,99,235,.09), rgba(13,148,136,.09));}
.panel p {margin: 0 0 1rem; line-height: 1.55;}
.barrow {display: grid; grid-template-columns: 10rem 1fr 3.2rem; align-items: center; gap: .7rem; margin: .55rem 0;}
.barl {font-size: .9rem; opacity: .9; font-weight: 600;}
.barv {font-weight: 800; text-align: right; font-size: .92rem;}
.track {height: 12px; border-radius: 99px; background: rgba(128,128,128,.25); overflow: hidden;}
.fill {height: 100%; border-radius: 99px;}
.fgrid {display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem;}
.fcard {border: 1px solid rgba(128,128,128,.25); border-radius: 18px; padding: 1.1rem 1.2rem; background: rgba(128,128,128,.06);}
.fcard h4 {margin: 0 0 .2rem; font-size: 1rem; font-weight: 800;}
.fcard small {display: block; opacity: .65; margin-bottom: .5rem;}
.fcard.static .fill {background: linear-gradient(90deg, var(--g1), var(--g2));}
.sens {width: 100%; border-collapse: collapse; margin-top: .4rem; font-size: .92rem;}
.sens th, .sens td {padding: .55rem .7rem; text-align: left; border-bottom: 1px solid rgba(128,128,128,.25);}
.sens th {font-size: .78rem; opacity: .7; text-transform: none;}
.sens tr.you td:first-child::after {content: " (your input)"; opacity: .65; font-weight: 600;}
@media (max-width: 560px) {.barrow {grid-template-columns: 1fr 3.2rem;} .barl {grid-column: 1 / -1;}}

.habits {display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: .8rem; margin-top: .9rem;}
.habit {padding: .8rem 1rem; border-radius: 14px; background: rgba(128,128,128,.10); font-size: .9rem; line-height: 1.45;}
.habit b {display: block; margin-bottom: .1rem;}
.focus {display: flex; gap: .8rem; align-items: flex-start; margin: .55rem 0; line-height: 1.5;}
.focus .e {font-size: 1.4rem; line-height: 1.2;}
.focus b {display: block;}
.tip {margin-top: .9rem; padding: .7rem .9rem; border-radius: 12px; font-size: .88rem; line-height: 1.45; background: rgba(128,128,128,.12);}
.tip b {display: block; font-size: .78rem; opacity: .7; margin-bottom: .15rem;}

/* ---------- nutrition guide ---------- */
.plate {display: flex; align-items: center; gap: 1.4rem; flex-wrap: wrap; padding: 1.1rem 1.3rem; margin-bottom: 1rem; border-radius: 18px;
  border: 1px solid rgba(128,128,128,.25); background: rgba(128,128,128,.07);}
.disc {flex: none; width: 118px; height: 118px; border-radius: 50%; border: 6px solid rgba(128,128,128,.35);
  background: conic-gradient(#16A34A 0 50%, #F59E0B 0 75%, #2563EB 0 100%);}
.plate ul {list-style: none; margin: 0; padding: 0; flex: 1 1 220px;}
.plate li {display: flex; align-items: center; gap: .6rem; margin: .35rem 0; font-size: .93rem; line-height: 1.35; font-weight: 600;}
.plate li i {flex: none; width: 14px; height: 14px; border-radius: 4px;}
.plate h4 {margin: 0 0 .4rem; font-size: 1rem; font-weight: 800;}
.fhead {display: flex; align-items: center; gap: .7rem; margin: .2rem 0 .6rem; padding: .7rem .9rem; border-radius: 14px;
  background: linear-gradient(120deg, color-mix(in srgb, var(--g1) 18%, transparent), color-mix(in srgb, var(--g2) 10%, transparent));
  border: 1px solid rgba(128,128,128,.25);}

.note {margin-top: 1.3rem; font-size: .84rem; opacity: .68; line-height: 1.5;}
.foot {margin-top: 2.4rem; padding-top: 1.1rem; border-top: 1px solid rgba(128,128,128,.25); text-align: center; font-size: .8rem; opacity: .65; line-height: 1.6;}
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
    score, lvl = round(p * 100), level(p)
    return (f'<div class="rcard {kind}"><div class="rhead"><span class="ico">{icon}</span>'
            f'<div><div class="rtitle">{title}</div><div class="rsub">Model score</div></div>'
            f'<span class="pill" style="--c:{COLORS[lvl]}">{lvl}</span></div>'
            f'<div class="ring" style="--p:{score};--c:{COLORS[lvl]}"><div class="ringtxt"><b>{score}</b><span>out of 100</span></div></div>'
            f'<div class="lvl" style="--c:{COLORS[lvl]}">{lvl} risk category</div>'
            f'<div class="rtext">{MEANING[lvl]}</div>'
            f'<div class="tip"><b>What you can do</b>{TIPS[key][lvl]}</div></div>')


def quote_html(text, author):
    return f'<div class="quote">&ldquo;{text}&rdquo;<small>{author}</small></div>'


def bar(label, p):
    pct = round(p * 100)
    return (f'<div class="barrow"><span class="barl">{label}</span>'
            f'<div class="track"><div class="fill" style="width:{pct}%;background:{COLORS[level(p)]}"></div></div>'
            f'<span class="barv">{pct}</span></div>')


def label_of(n):
    return FRIENDLY.get(n, n.replace("_", " ").capitalize())


def importance_card(title, kind, model, top=5):
    imp = pd.Series(model.feature_importances_, index=model.feature_names_in_).sort_values(ascending=False).head(top)
    mx = imp.max() or 1
    rows = "".join(
        f'<div class="barrow"><span class="barl">{label_of(n)}</span>'
        f'<div class="track"><div class="fill" style="width:{v / mx * 100:.0f}%"></div></div>'
        f'<span class="barv">{v * 100:.0f}%</span></div>' for n, v in imp.items())
    return f'<div class="fcard static {kind}"><h4>{title}</h4><small>Most relied-on inputs overall</small>{rows}</div>'


def drivers(model, vals, ref, top=4):
    """Swap each of the user's values for a typical reference value and see how the model score changes."""
    base = model.predict_proba(pd.DataFrame([vals]))[0, 1]
    out = []
    for k in vals:
        if k == "sex":
            continue
        alt = dict(vals)
        alt[k] = ref[k]
        out.append((k, base - model.predict_proba(pd.DataFrame([alt]))[0, 1]))
    out = [x for x in sorted(out, key=lambda x: -abs(x[1])) if abs(x[1]) >= 0.005]
    return out[:top]


def drivers_card(title, kind, items):
    if not items:
        body = '<div class="rsub">No single value stood out compared with a typical profile.</div>'
    else:
        mx = max(abs(d) for _, d in items) or 1
        body = "".join(
            f'<div class="barrow"><span class="barl">{"▲" if d > 0 else "▼"} {label_of(k)}</span>'
            f'<div class="track"><div class="fill" style="width:{abs(d) / mx * 100:.0f}%;background:{"#DC2626" if d > 0 else "#16A34A"}"></div></div>'
            f'<span class="barv">{d * 100:+.0f}</span></div>' for k, d in items)
    return f'<div class="fcard {kind}"><h4>{title}</h4><small>▲ raises your score · ▼ lowers it (points out of 100)</small>{body}</div>'


def plate_html():
    return ('<div class="plate"><div class="disc"></div><ul><h4>The healthy plate</h4>'
            '<li><i style="background:#16A34A"></i>Half: vegetables and salad</li>'
            '<li><i style="background:#F59E0B"></i>One quarter: whole grains, millets or roti</li>'
            '<li><i style="background:#2563EB"></i>One quarter: dal, egg, fish or lean meat</li></ul></div>')


def food_focus(lvl_d, lvl_h, glucose, sbp, chol):
    out = []
    if ORDER[lvl_d] >= 1 or glucose > 120:
        out.append(("🩸", "For blood sugar",
                    "Cut sugary drinks and sweets, and choose whole grains, millets and dal over white rice and maida."))
    if ORDER[lvl_h] >= 1 or sbp >= 130 or chol >= 200:
        out.append(("❤️", "For your heart",
                    "Eat less salt and fried food, and add vegetables, fruit, nuts and oily fish."))
    if not out:
        out.append(("🥗", "Keep it balanced",
                    "Keep eating a mix of vegetables, whole grains, protein and fruit, and limit sweets and fried food."))
    return out


def bmi_category(v):
    return "Underweight" if v < 18.5 else "Normal" if v < 25 else "Overweight" if v < 30 else "Obese range"


def with_unknown(col, label, mn, mx, typical, key, help=None):
    """Number input plus an 'I don't know' checkbox. Returns (value, was_assumed)."""
    box = col.container()
    unknown = col.checkbox("I don't know", key=f"{key}_unk")
    val = box.number_input(label, mn, mx, typical, help=help, disabled=unknown, key=key)
    return (typical, True) if unknown else (val, False)


def reset_form():
    st.session_state.clear()


# ------------------------------------------------------------------ header
st.markdown(
    '<div class="brand"><span class="logo">🛡️</span><div><b>HealthGuard AI</b><small>Disease Risk Prediction System</small></div>'
    '<span class="bt">AI-powered screening</span></div>'
    '<div class="hero"><span class="tag"><i></i>AI-powered health screening</span>'
    '<h1>Understand your health. Take action early.</h1>'
    '<p>Answer a few health questions to explore your estimated diabetes and heart disease risk.</p>'
    '<svg class="ecg" viewBox="0 0 420 44" preserveAspectRatio="none"><path d="M0 24 H70 L82 24 L92 8 L104 40 '
    'L114 24 H190 L202 24 L212 6 L224 42 L234 24 H310 L322 24 L332 10 L344 38 L354 24 H420"/></svg></div>'
    '<div class="chips"><span class="chip c1">🩸 Diabetes screening</span><span class="chip c2">❤️ Heart health screening</span>'
    '<span class="chip c3">Educational tool · Not a medical diagnosis</span></div>',
    unsafe_allow_html=True,
)

if "quote" not in st.session_state:
    st.session_state["quote"] = random.choice(QUOTES)
st.markdown(quote_html(*st.session_state["quote"]), unsafe_allow_html=True)
st.write("")

tab_basic, tab_lab, tab_food = st.tabs(["①  Health details", "②  Lab values", "③  Nutrition guide"])
assumed = []

# ------------------------------------------------------------------ inputs
with tab_basic:
    st.markdown('<div class="help">Things you can find out at home or from a quick check-up.</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec">👤 Personal information</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    age = c1.number_input("Age (years)", 1, 100, 40)
    sex = c2.selectbox("Sex", ["Male", "Female"])
    height = c3.number_input("Height (cm)", 100, 220, 170)
    weight = c4.number_input("Weight (kg)", 25, 250, 70)
    bmi = round(weight / ((height / 100) ** 2), 1)
    st.markdown(f'<div class="bmi">Your BMI: {bmi} · {bmi_category(bmi)}</div>', unsafe_allow_html=True)
    if sex == "Female":
        preg = st.number_input("Number of pregnancies", 0, 20, 0)
    else:
        preg = 0

    st.markdown('<div class="sec">🩺 Vital signs</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    sbp = c1.number_input("BP top number (mmHg)", 80, 220, 120, help="Systolic pressure. A typical healthy reading is around 120.")
    dbp = c2.number_input("BP bottom number (mmHg)", 30, 140, 80, help="Diastolic pressure. A typical healthy reading is around 80.")
    glucose, a1 = with_unknown(c3, "Fasting blood sugar (mg/dL)", 40, 400, TYPICAL["glucose"], "glucose",
                               help="Blood sugar after not eating for 8 hours. Above 120 counts as high here.")
    chol, a2 = with_unknown(c4, "Cholesterol (mg/dL)", 100, 600, TYPICAL["chol"], "chol",
                            help="Total cholesterol from a blood test.")
    if a1:
        assumed.append("fasting blood sugar")
    if a2:
        assumed.append("cholesterol")

    st.markdown('<div class="sec">🫀 Chest symptoms</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    cp_label = c1.selectbox("Chest pain level", list(CHEST_PAIN), index=0)
    cp = CHEST_PAIN[cp_label]
    c1.caption(f"Medical type: {CHEST_PAIN_TYPE[cp_label]}")
    exang = c2.selectbox("Chest pain during exercise?", ["No", "Yes"])

with tab_lab:
    st.markdown('<div class="notice"><b>Optional clinical measurements</b>Enter these values only if they come from a '
                'medical test or report. If you do not have them, leave the box below unticked: typical values will be '
                'used and your results will be clearly marked as using typical values.</div>', unsafe_allow_html=True)
    have_lab = st.checkbox("I have lab report values", key="have_lab")
    if have_lab:
        a, b, c = st.columns(3)
        insulin = a.number_input("Insulin level (mu U/ml)", 0, 900, TYPICAL["insulin"],
                                 help="Blood insulin from a lab test.")
        skin = b.number_input("Skin fold thickness (mm)", 5, 100, TYPICAL["skin"],
                              help="Measured with a skin-fold caliper.")
        dpf = c.number_input("Diabetes family history score", 0.05, 2.5, TYPICAL["dpf"],
                             help="A score that reflects diabetes in your relatives. Typical value is about 0.4.")
        hr = a.number_input("Highest heart rate in exercise test", 60, 220, TYPICAL["hr"])
        ecg_label = b.selectbox("Resting ECG result", list(ECG), index=0)
        slope_label = c.selectbox("Exercise ECG slope", list(SLOPE), index=1)
        oldpeak = a.number_input("ST depression (oldpeak)", 0.0, 7.0, TYPICAL["oldpeak"],
                                 help="A measurement from an exercise ECG test.")
        ca = b.selectbox("Blocked vessels seen on scan (0 to 3)", [0, 1, 2, 3])
        thal_label = c.selectbox("Thalassemia blood test", list(THAL), index=0)
    else:
        st.info("Using typical values for the lab measurements. Tick the box above if you have your own report values.")
        insulin, skin, dpf, hr = TYPICAL["insulin"], TYPICAL["skin"], TYPICAL["dpf"], TYPICAL["hr"]
        ecg_label, slope_label, oldpeak = TYPICAL["ecg"], TYPICAL["slope"], TYPICAL["oldpeak"]
        ca, thal_label = TYPICAL["ca"], TYPICAL["thal"]
        assumed.append("lab values (insulin, skin fold, family history score, ECG results, scan results)")
    ecg, slope, thal = ECG[ecg_label], SLOPE[slope_label], THAL[thal_label]

with tab_food:
    st.markdown('<div class="help">What to eat and what to limit for a healthy sugar level and a healthy heart. '
                'This is general guidance, not a personal diet plan.</div>', unsafe_allow_html=True)
    st.markdown(plate_html(), unsafe_allow_html=True)
    fc1, fc2 = st.columns(2)
    for col, key in ((fc1, "diabetes"), (fc2, "heart")):
        f = FOOD[key]
        with col:
            st.markdown(f'<div class="fhead {f["kind"]}" style="--g1:{"#2563EB" if key == "diabetes" else "#E11D48"};'
                        f'--g2:{"#06B6D4" if key == "diabetes" else "#F97316"}"><span class="ico" style="--g1:{"#2563EB" if key == "diabetes" else "#E11D48"};'
                        f'--g2:{"#06B6D4" if key == "diabetes" else "#F97316"}">{f["icon"]}</span>'
                        f'<div class="rtitle">{f["title"]}</div></div>', unsafe_allow_html=True)
            for cls, name, items in f["groups"]:
                with st.expander(f"{GROUP_ICON[cls]} {name}", expanded=(cls == "good")):
                    st.markdown("\n".join(f"- {i}" for i in items))
    st.caption("Eat regular meals, watch portion sizes, drink plenty of water and stay active. "
               "If you are pregnant, have kidney disease or take medicines for diabetes or blood pressure, "
               "follow your doctor's or dietitian's advice.")

st.write("")
go = st.button("Check my risk", type="primary")

# ------------------------------------------------------------------ results
if go:
    with st.spinner("Analyzing your inputs..."):
        fbs = int(glucose > 120)
        d_vals = {"Pregnancies": preg, "Glucose": glucose, "BloodPressure": dbp, "SkinThickness": skin,
                  "Insulin": insulin, "BMI": bmi, "DiabetesPedigreeFunction": dpf, "Age": age}
        h_vals = {"age": age, "sex": int(sex == "Male"), "chest_pain_type": cp, "resting_bp": sbp,
                  "cholesterol": chol, "fasting_blood_sugar": fbs, "resting_ecg": ecg, "max_heart_rate": hr,
                  "exercise_angina": int(exang == "Yes"), "oldpeak": oldpeak, "slope": slope,
                  "num_major_vessels": ca, "thalassemia": thal}
        p_d = models["diabetes"].predict_proba(pd.DataFrame([d_vals]))[0, 1]
        p_h = models["heart"].predict_proba(pd.DataFrame([h_vals]))[0, 1]
        p_lo = models["heart"].predict_proba(pd.DataFrame([{**h_vals, "fasting_blood_sugar": 0}]))[0, 1]
        p_hi = models["heart"].predict_proba(pd.DataFrame([{**h_vals, "fasting_blood_sugar": 1}]))[0, 1]
        drv_d = drivers(models["diabetes"], d_vals, REF_D)
        drv_h = drivers(models["heart"], h_vals, REF_H)

    lvl_d, lvl_h = level(p_d), level(p_h)
    tone = max(lvl_d, lvl_h, key=lambda x: ORDER[x])
    icon, head, sub = HEADLINE[tone]
    assumed_txt = "; ".join(assumed)

    # 1. Prediction summary + 2. Risk indicators
    st.markdown('<div class="section-title">Your screening results</div>'
                '<div class="section-sub">Results based on the health information you entered.</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="overall" style="--c:{COLORS[tone]}"><div class="big">{icon}</div><div><h3>{head}</h3>'
        f'<p>{sub}</p><small>Diabetes: {lvl_d} · Heart disease: {lvl_h}</small></div></div>',
        unsafe_allow_html=True,
    )
    if assumed:
        st.markdown(f'<div class="assume"><b>Typical values were used for:</b> {assumed_txt}. '
                    'These are not your own measurements, so treat the results as a rough guide.</div>',
                    unsafe_allow_html=True)
    st.write("")
    st.markdown(f'<div class="rgrid">{result_card("Diabetes", p_d, "d", "🩸", "diabetes")}'
                f'{result_card("Heart disease", p_h, "h", "❤️", "heart")}</div>', unsafe_allow_html=True)

    # 3. Model explanation
    st.markdown('<div class="section-title">What influenced your result?</div>'
                '<div class="section-sub">We swapped each of your values for a typical reference value and measured how '
                'the model score changed. These explanations come from the trained Random Forest models.</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="fgrid">{drivers_card("Diabetes score", "d", drv_d)}'
                f'{drivers_card("Heart disease score", "h", drv_h)}</div>', unsafe_allow_html=True)

    delta = (p_hi - p_lo) * 100
    if delta > 0.5:
        reading = f"For this profile, the heart score is about {delta:.0f} points higher with high blood sugar."
    elif delta < -0.5:
        reading = "For this profile, the heart score is not higher with high blood sugar."
    else:
        reading = "For this profile, blood sugar makes little difference to the heart score."
    st.markdown('<div class="section-title">Model sensitivity: blood sugar and heart score</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel">'
        '<p>We compare two inputs while keeping everything else you entered the same, to show how the heart model '
        f'responds to fasting blood sugar. {reading} This compares model outputs. It is not proof that one condition causes the other.</p>'
        '<table class="sens"><tr><th>Scenario</th><th>Fasting blood sugar</th><th>Heart model score</th><th>Change</th></tr>'
        f'<tr class="{"you" if fbs == 0 else ""}"><td>Baseline</td><td>120 mg/dL or below</td><td>{p_lo * 100:.0f} / 100</td><td>-</td></tr>'
        f'<tr class="{"you" if fbs == 1 else ""}"><td>Modified</td><td>Above 120 mg/dL</td><td>{p_hi * 100:.0f} / 100</td><td>{delta:+.0f} points</td></tr></table>'
        f'{bar("Normal blood sugar", p_lo)}{bar("High blood sugar", p_hi)}</div>',
        unsafe_allow_html=True,
    )

    # 4. Health recommendations
    focus = food_focus(lvl_d, lvl_h, glucose, sbp, chol)
    st.markdown('<div class="section-title">Health recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel">' + "".join(
        f'<div class="focus"><span class="e">{e}</span><div><b>{t}</b>{d}</div></div>' for e, t, d in focus)
        + '<div class="habits">'
          '<div class="habit"><b>🚶 Stay active</b>Aim for at least 150 minutes of moderate activity a week.</div>'
          '<div class="habit"><b>🚭 Avoid tobacco</b>Smoking raises heart risk. Ask for help if you want to quit.</div>'
          '<div class="habit"><b>😴 Sleep well</b>Most adults need about 7 to 9 hours of sleep.</div>'
          '<div class="habit"><b>🩺 Regular check-ups</b>Check blood pressure, sugar and cholesterol regularly.</div></div>'
          '<div class="note" style="margin-top:.8rem">See the full list of foods to eat and limit in the '
          '<b>③ Nutrition guide</b> tab above.</div></div>', unsafe_allow_html=True)

    with st.expander("How does the model work?"):
        st.markdown(f'<div class="fgrid">{importance_card("Diabetes model", "d", models["diabetes"])}'
                    f'{importance_card("Heart model", "h", models["heart"])}</div>', unsafe_allow_html=True)
        st.caption("A Random Forest combines many decision trees. These are the inputs it relies on most across all its "
                   "training data, not for you alone.")

    # 5. Report
    summary = (
        "HEALTHGUARD AI - DISEASE RISK SCREENING SUMMARY\n\n"
        f"Diabetes model score: {p_d * 100:.0f}/100 ({lvl_d})\n"
        f"Heart disease model score: {p_h * 100:.0f}/100 ({lvl_h})\n"
        f"Overall: {head}\n\n"
        "Food focus:\n" + "\n".join(f"- {t}: {d}" for _, t, d in focus) + "\n\n"
        f"Inputs: age {age}, {sex}, BMI {bmi}, BP {sbp}/{dbp} mmHg, fasting sugar {glucose} mg/dL, cholesterol {chol} mg/dL.\n"
        + (f"Typical values were used for: {assumed_txt}.\n" if assumed else "")
        + "\nThis is an educational screening tool, not a medical diagnosis. Model scores are not exact probabilities. "
          "Please consult a doctor."
    )
    st.write("")
    b1, b2 = st.columns(2)
    with b1:
        st.download_button("Download my summary", summary, file_name="healthguard_summary.txt")
    with b2:
        st.button("Reset form", on_click=reset_form)

    st.markdown(quote_html(*RESULT_QUOTES[tone]), unsafe_allow_html=True)
    st.markdown(
        '<div class="note">This is an educational screening score from a Random Forest model, not a diagnosis and not the '
        'exact chance of getting a disease. The diabetes and heart models are trained on separate datasets, so the '
        'blood sugar comparison shows an association the model learned, not proof of cause. Please consult a doctor for medical advice.</div>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="foot"><b>HealthGuard AI</b> · Disease Risk Prediction System · Random Forest · Streamlit · v1.0<br>'
            'Artificial Intelligence and Data Science · HNN College of Engineering<br>'
            'Educational screening tool, not a medical diagnosis. This app has no accounts or database and does not '
            'store your answers itself.</div>', unsafe_allow_html=True)
