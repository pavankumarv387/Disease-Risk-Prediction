import random
from datetime import datetime, timedelta, timezone

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Disease Risk Prediction", page_icon="🩺", layout="centered")

COLORS = {"Low": "#16A34A", "Moderate": "#D97706", "High": "#DC2626"}
CUT_LOW, CUT_HIGH = 0.33, 0.66

QUOTES = [
    ("Prevention is better than cure.", "Proverb"),
    ("Health is wealth.", "Proverb"),
    ("An apple a day keeps the doctor away.", "Proverb"),
    ("The greatest wealth is health.", "Virgil"),
    ("Early to bed and early to rise makes a man healthy, wealthy, and wise.", "Benjamin Franklin"),
]
MEANING = {
    "Low": "Few risk signs were found in the details you entered.",
    "Moderate": "Some risk signs are present. Monitor and consider a check-up.",
    "High": "Several risk signs are present. A medical check-up is advised.",
}

# ---------- health tips shown for each risk level ----------
TIPS = {
    "diabetes": {
        "Low": [
            ("🥗", "Keep a balanced diet with whole grains, vegetables, pulses and fruit."),
            ("🚶", "Stay active: aim for about 150 minutes of moderate exercise a week."),
            ("⚖️", "Maintain a healthy weight and BMI."),
            ("🩺", "Check your fasting blood sugar once a year."),
        ],
        "Moderate": [
            ("🥤", "Cut down on sugary drinks, sweets and refined carbohydrates (white rice, maida)."),
            ("🚶", "Walk briskly for 30 minutes on most days of the week."),
            ("⚖️", "If you are overweight, losing even 5-7% of body weight lowers risk."),
            ("🩺", "Get fasting glucose or HbA1c tested every 6 months."),
            ("😴", "Sleep 7-8 hours and manage stress; both affect blood sugar."),
        ],
        "High": [
            ("🩺", "Book a doctor visit soon for fasting glucose and HbA1c tests."),
            ("🥗", "Ask a doctor or dietitian for a meal plan; avoid sugary food and drinks."),
            ("📈", "Monitor your blood sugar if your doctor advises it."),
            ("⚠️", "Do not ignore symptoms: excess thirst, frequent urination, blurred vision, slow-healing wounds."),
            ("🚶", "Stay active with light exercise such as walking, once your doctor agrees."),
        ],
    },
    "heart": {
        "Low": [
            ("🥦", "Eat a heart-friendly diet: more fruit and vegetables, less salt and fried food."),
            ("🏃", "Stay physically active most days of the week."),
            ("🚭", "Avoid tobacco and limit alcohol."),
            ("🩺", "Check blood pressure and cholesterol once a year."),
        ],
        "Moderate": [
            ("🧂", "Reduce salt, saturated fat and processed food."),
            ("🏃", "Do 30 minutes of brisk activity on most days."),
            ("🚭", "Quit smoking and limit alcohol if you use either."),
            ("🩺", "Check blood pressure and cholesterol every 6 months."),
            ("🧘", "Manage stress with rest, breathing exercises or hobbies."),
        ],
        "High": [
            ("🩺", "See a doctor soon for an ECG, blood pressure and cholesterol check."),
            ("🚨", "Seek urgent care for chest pain, breathlessness, fainting or pain spreading to the arm or jaw."),
            ("🛑", "Avoid heavy exertion until a doctor has checked your heart."),
            ("💊", "Take any prescribed medicines regularly and do not stop them on your own."),
            ("🧂", "Cut salt, fried food and tobacco right away."),
        ],
    },
}

# ---------- food guide: (text, vegetarian replacement or None to hide for vegetarians) ----------
SAME = "same"
FOOD = {
    "diabetes": {
        "Eat more": [
            ("Non-starchy vegetables: spinach, beans, gourds, cabbage, cauliflower", SAME),
            ("Dal, sprouts, chickpeas and other legumes", SAME),
            ("Whole grains and millets: ragi, jowar, oats, brown rice", SAME),
            ("Lean protein: fish, eggs, skinless chicken, paneer", "Lean protein: paneer, tofu, soy chunks"),
            ("Unsweetened curd or buttermilk, and a small handful of nuts", SAME),
        ],
        "In moderation": [
            ("Rice, chapati and bread: keep portions small", SAME),
            ("Whole fruits (not juice): one serving at a time", SAME),
            ("Potato, sweet potato and other starchy vegetables", SAME),
            ("Honey and jaggery: they are still sugar", SAME),
        ],
        "Limit or avoid": [
            ("Sugary drinks, packaged juices, sweets and desserts", SAME),
            ("Maida, white bread and bakery items", SAME),
            ("Deep-fried snacks and packaged foods", SAME),
            ("Extra sugar in tea and coffee, and alcohol", SAME),
        ],
    },
    "heart": {
        "Eat more": [
            ("A variety of vegetables and fruits every day", SAME),
            ("Whole grains, oats and millets", SAME),
            ("Dal, beans and other legumes", SAME),
            ("Oily fish such as sardine or mackerel, about twice a week", None),
            ("Unsalted nuts and seeds such as flaxseed", SAME),
        ],
        "In moderation": [
            ("Cooking oil: use small amounts of groundnut, sunflower or olive oil", SAME),
            ("Dairy: choose low-fat milk and curd", SAME),
            ("Eggs and skinless chicken", None),
            ("Tea and coffee without much sugar", SAME),
        ],
        "Limit or avoid": [
            ("Salt, pickles, papad and salty snacks (aim for under 5 g of salt a day)", SAME),
            ("Deep-fried food such as samosa, pakora and chips", SAME),
            ("Ghee, butter, vanaspati and too much coconut oil", SAME),
            ("Red and processed meat, sugary drinks and alcohol", "Sugary drinks and alcohol"),
        ],
    },
}
FOOD_COLORS = {"Eat more": "#16A34A", "In moderation": "#D97706", "Limit or avoid": "#DC2626"}

# Plain-language chest pain choices; each maps to the dataset's code (1-4) used by the trained models.
CHEST_PAIN = {
    "No chest pain": 4,
    "Mild or occasional discomfort": 3,
    "Moderate pain or tightness": 2,
    "Severe pressure or pain (may spread to arm or jaw)": 1,
}
# Family history shown in words; each maps to an approximate diabetes pedigree function value.
FAMILY = {
    "No known diabetes in the family": 0.2,
    "A parent or sibling has diabetes": 0.5,
    "Several close relatives have diabetes": 1.0,
}
ECG_LABELS = {0: "Normal", 1: "ST-T wave abnormality", 2: "Left ventricular hypertrophy"}
SLOPE_LABELS = {1: "Upsloping", 2: "Flat", 3: "Downsloping"}
THAL_LABELS = {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"}
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
h1, h2, h3, p, label, button, input, textarea, li {font-family: 'Manrope', system-ui, sans-serif;}
#MainMenu, footer {visibility: hidden;}
.stApp {
  background-image:
    radial-gradient(circle at 0% 0%, rgba(37,99,235,.13), transparent 40%),
    radial-gradient(circle at 100% 100%, rgba(13,148,136,.13), transparent 42%);
}
.block-container {max-width: 800px; padding-top: 1.5rem; padding-bottom: 3rem;}
div[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 18px;}
label, .stRadio label {font-weight: 600;}

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
.hero h1 {color: #fff !important; font-size: 2.3rem; font-weight: 800; letter-spacing: -0.02em; margin: 0 0 .45rem; padding: 0; line-height: 1.1;}
.hero p {color: #fff !important; margin: 0; opacity: .9; max-width: 34rem; line-height: 1.6; font-size: 1.02rem;}
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
.c4 {background: linear-gradient(90deg, #475569, #64748B);}
.steps {display: flex; gap: .6rem; flex-wrap: wrap; margin: 1.1rem 0 .3rem;}
.stp {flex: 1 1 150px; display: flex; align-items: center; gap: .6rem; padding: .6rem .8rem; border-radius: 14px;
  border: 1px solid rgba(128,128,128,.25); background: rgba(128,128,128,.07); font-weight: 700; font-size: .9rem;}
.stp b {display: inline-grid; place-items: center; width: 1.7rem; height: 1.7rem; border-radius: 50%; color: #fff; font-size: .85rem;
  background: linear-gradient(135deg, #1D4ED8, #0D9488);}

/* ---------- quote ---------- */
.quote {margin: 1.1rem 0 .5rem; padding: .95rem 1.2rem 1rem 1.4rem; border-radius: 14px; border-left: 5px solid #2563EB;
  background: linear-gradient(120deg, rgba(37,99,235,.12), rgba(13,148,136,.12));
  font-style: italic; line-height: 1.5; font-size: 1.03rem;}
.quote small {display: block; font-style: normal; opacity: .75; margin-top: .3rem; font-size: .88rem; font-weight: 600;}

/* ---------- form sections ---------- */
.step {display: flex; align-items: center; gap: .8rem; margin: .1rem 0 .9rem;}
.stepno {display: inline-grid; place-items: center; width: 2.3rem; height: 2.3rem; border-radius: 12px; font-size: 1.1rem;
  background: linear-gradient(135deg, #1D4ED8, #0D9488); box-shadow: 0 8px 18px -8px rgba(29,78,216,.8); flex: none;}
.steptitle {font-size: 1.15rem; font-weight: 800; line-height: 1.15;}
.stepsub {font-size: .9rem; opacity: .8; margin-top: .1rem;}
.hint {padding: .65rem .9rem; border-radius: 12px; font-size: .9rem; line-height: 1.5; margin: .3rem 0 .8rem;
  background: rgba(37,99,235,.09); border: 1px dashed rgba(37,99,235,.45);}
.subhead {font-weight: 800; margin: .9rem 0 .2rem; font-size: 1rem;}

/* ---------- BMI ---------- */
.bmi {margin: .4rem 0 .6rem; padding: 1.1rem 1.2rem 1rem; border-radius: 16px; border: 1px solid rgba(37,99,235,.28);
  background: linear-gradient(135deg, rgba(37,99,235,.09), rgba(13,148,136,.09));}
.bmitop {display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;}
.bmil {font-size: .85rem; opacity: .8; font-weight: 700;}
.bmiv {font-size: 2.1rem; font-weight: 800; line-height: 1.05;}
.bmibar {position: relative; height: 12px; border-radius: 99px;
  background: linear-gradient(90deg, #38BDF8 0 14%, #16A34A 14% 40%, #D97706 40% 60%, #DC2626 60% 100%);}
.bmibar span, .scale span {position: absolute; top: 50%; width: 20px; height: 20px; border-radius: 50%; background: #fff;
  border: 4px solid #0B1F44; transform: translate(-50%, -50%); box-shadow: 0 2px 8px rgba(0,0,0,.35);}
.bmiscale {display: flex; justify-content: space-between; gap: .4rem; flex-wrap: wrap; margin-top: .6rem;}
.bmiscale small {font-size: .78rem; opacity: .8; font-weight: 600;}

.stButton > button, .stDownloadButton > button {width: 100%; border-radius: 12px; font-weight: 800; padding: .85rem 1rem; font-size: 1.06rem; transition: transform .15s ease, box-shadow .15s ease;}
.stButton > button[kind="primary"], .stButton button[data-testid="stBaseButton-primary"] {
  border: none; background-image: linear-gradient(100deg, #1D4ED8, #0D9488); background-color: #1D4ED8; color: #fff;
  box-shadow: 0 14px 28px -14px rgba(29,78,216,.9);}
.stButton > button[kind="primary"]:hover, .stButton button[data-testid="stBaseButton-primary"]:hover {
  transform: translateY(-2px); color: #fff; box-shadow: 0 18px 32px -14px rgba(13,148,136,.9);}

/* ---------- results ---------- */
.section-title {font-size: 1.25rem; font-weight: 800; margin: 2.2rem 0 1rem; letter-spacing: -0.01em; display: flex; align-items: center; gap: .6rem;}
.section-title::before {content: ""; width: 5px; height: 1.3rem; border-radius: 4px; background: linear-gradient(180deg, #1D4ED8, #0D9488);}
.stamp {display: inline-block; padding: .35rem .85rem; border-radius: 99px; font-size: .88rem; font-weight: 700; margin: -.4rem 0 .9rem;
  background: rgba(37,99,235,.12); border: 1px solid rgba(37,99,235,.4);}
.cmp {margin: 1rem 0 0; padding: .9rem 1.1rem; border-radius: 14px; border: 1px solid rgba(128,128,128,.25); background: rgba(128,128,128,.06);
  line-height: 1.6; font-size: .93rem;}
.legend {display: flex; gap: .5rem; flex-wrap: wrap; margin: -.3rem 0 .6rem; font-size: .82rem; font-weight: 700;}
.legend span {padding: .25rem .7rem; border-radius: 99px; color: var(--c); border: 1px solid var(--c); background: rgba(128,128,128,.1);}
.scorenote {font-size: .88rem; opacity: .8; margin: 0 0 1rem; line-height: 1.5;}
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
.rsub {font-size: .85rem; opacity: .75; margin-top: .15rem;}
.pill {margin-left: auto; padding: .28rem .75rem; border-radius: 99px; font-size: .8rem; font-weight: 800; color: var(--c);
  background: rgba(128,128,128,.15); background: color-mix(in srgb, var(--c) 16%, transparent);
  border: 1px solid color-mix(in srgb, var(--c) 45%, transparent);}
.ring {position: relative; width: 150px; height: 150px; margin: .2rem auto .7rem;}
.ring::before {content: ""; position: absolute; inset: 0; border-radius: 50%;
  background: conic-gradient(var(--c) calc(var(--p) * 1%), rgba(128,128,128,.25) 0);
  -webkit-mask: radial-gradient(farthest-side, transparent calc(100% - 15px), #000 calc(100% - 14px));
  mask: radial-gradient(farthest-side, transparent calc(100% - 15px), #000 calc(100% - 14px));}
.ringtxt {position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center;}
.ringtxt b {font-size: 2.05rem; font-weight: 800; line-height: 1;}
.ringtxt span {font-size: .85rem; font-weight: 700; margin-top: .25rem; opacity: .8;}
.scale {position: relative; height: 10px; border-radius: 99px; margin: .9rem .4rem .35rem;
  background: linear-gradient(90deg, #16A34A 0 33%, #D97706 33% 66%, #DC2626 66% 100%);}
.scale span {width: 16px; height: 16px; border-width: 3px;}
.scalel {display: flex; justify-content: space-between; margin: 0 .4rem .8rem;}
.scalel small {font-size: .75rem; opacity: .8; font-weight: 700;}
.rtext {text-align: center; font-size: .92rem; line-height: 1.45; opacity: .9;}
.border {margin-top: .6rem; text-align: center; font-size: .84rem; font-weight: 700; color: #D97706;}

/* ---------- health tips ---------- */
.alert {margin: 0 0 1rem; padding: .8rem 1rem; border-radius: 14px; border-left: 6px solid #DC2626; line-height: 1.5; font-size: .93rem;
  background: rgba(220,38,38,.1);}
.tgrid {display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem;}
.tcard {position: relative; overflow: hidden; padding: 1.3rem 1.2rem 1.1rem; border-radius: 20px; border: 1px solid rgba(128,128,128,.25);
  background: rgba(128,128,128,.06);}
.tcard::before {content: ""; position: absolute; left: 0; right: 0; top: 0; height: 6px; background: var(--c);}
.thead {display: flex; align-items: center; gap: .6rem; margin-bottom: .9rem;}
.thead b {font-size: 1.02rem; font-weight: 800;}
.titem {display: flex; gap: .7rem; align-items: flex-start; padding: .55rem .7rem; margin: .4rem 0; border-radius: 12px;
  background: rgba(128,128,128,.09); line-height: 1.45; font-size: .93rem;}
.titem .te {font-size: 1.15rem; line-height: 1.3;}
.titem.extra {border-left: 4px solid var(--c);}

/* ---------- food guide ---------- */
.plate {display: flex; gap: 1.4rem; align-items: center; flex-wrap: wrap; padding: 1.1rem 1.3rem; border-radius: 20px;
  border: 1px solid rgba(128,128,128,.25); background: rgba(128,128,128,.06); margin-bottom: .8rem;}
.pie {width: 120px; height: 120px; border-radius: 50%; flex: none;
  background: conic-gradient(#16A34A 0 50%, #D97706 50% 75%, #2563EB 75% 100%);}
.plate ul {list-style: none; margin: 0; padding: 0; line-height: 1.9; font-size: .95rem;}
.plate li::before {content: ""; display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: .6rem; background: var(--c);}
.fgroup {margin: .6rem 0; padding: .8rem 1rem; border-radius: 14px; border-left: 5px solid var(--c);
  background: rgba(128,128,128,.08);}
.fgroup h5 {margin: 0 0 .4rem; font-size: .98rem; font-weight: 800; color: var(--c);}
.fgroup ul {margin: 0; padding-left: 1.2rem; line-height: 1.6; font-size: .93rem;}
.fnote {font-size: .88rem; opacity: .8; margin-top: .7rem; line-height: 1.5;}

.panel {border: 1px solid rgba(37,99,235,.28); border-radius: 20px; padding: 1.25rem 1.35rem;
  background: linear-gradient(135deg, rgba(37,99,235,.09), rgba(13,148,136,.09));}
.panel p {margin: 0 0 1rem; line-height: 1.55;}
.barrow {display: grid; grid-template-columns: 9.5rem 1fr 3rem; align-items: center; gap: .7rem; margin: .55rem 0;}
.barl {font-size: .92rem; opacity: .95; font-weight: 600;}
.barv {font-weight: 800; text-align: right; font-size: .92rem;}
.track {height: 12px; border-radius: 99px; background: rgba(128,128,128,.25); overflow: hidden;}
.fill {height: 100%; border-radius: 99px;}
.fgrid {display: grid; grid-template-columns: repeat(auto-fit, minmax(270px, 1fr)); gap: 1rem;}
.fcard {border: 1px solid rgba(128,128,128,.25); border-radius: 18px; padding: 1.1rem 1.2rem; background: rgba(128,128,128,.06);}
.fcard h4 {margin: 0 0 .6rem; font-size: .98rem; font-weight: 800;}
.fcard .fill {background: linear-gradient(90deg, var(--g1), var(--g2));}
.fcard.d {--g1: #2563EB; --g2: #06B6D4;}
.fcard.h {--g1: #E11D48; --g2: #F97316;}
@media (max-width: 520px) {.barrow {grid-template-columns: 1fr 3rem;} .barl {grid-column: 1 / -1;} .hero h1 {font-size: 1.9rem;}}

.note {margin-top: 1.3rem; font-size: .88rem; opacity: .8; line-height: 1.5;}
.foot {margin-top: 2.2rem; padding-top: 1rem; border-top: 1px solid rgba(128,128,128,.25); text-align: center; font-size: .85rem; opacity: .75;}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def load_models():
    return {d: joblib.load(f"models/{d}_rf.joblib") for d in ("diabetes", "heart")}


models = load_models()


def level(p):
    return "Low" if p < CUT_LOW else "Moderate" if p < CUT_HIGH else "High"


def borderline(p):
    """Short note when a score sits close to a cut-off."""
    if abs(p - CUT_LOW) <= 0.05:
        return "Borderline: close to the Moderate range"
    if abs(p - CUT_HIGH) <= 0.05:
        return "Borderline: close to the High range"
    return ""


def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight", "#0EA5E9"
    if bmi < 25:
        return "Normal", "#16A34A"
    if bmi < 30:
        return "Overweight", "#D97706"
    return "Obese", "#DC2626"


def bmi_card(bmi):
    cat, col = bmi_category(bmi)
    pos = min(max((bmi - 15) / (40 - 15), 0), 1) * 100
    return (f'<div class="bmi"><div class="bmitop"><div><div class="bmil">Your BMI (calculated)</div>'
            f'<div class="bmiv">{bmi:.1f}</div></div><span class="pill" style="--c:{col}">{cat}</span></div>'
            f'<div class="bmibar"><span style="left:{pos:.1f}%"></span></div>'
            f'<div class="bmiscale"><small>Underweight &lt; 18.5</small><small>Normal 18.5-24.9</small>'
            f'<small>Overweight 25-29.9</small><small>Obese 30+</small></div></div>')


def section_header(icon, title, sub):
    return (f'<div class="step"><span class="stepno">{icon}</span><div><div class="steptitle">{title}</div>'
            f'<div class="stepsub">{sub}</div></div></div>')


def result_card(title, p, kind, icon):
    pct, lvl = round(p * 100), level(p)
    note = borderline(p)
    note_html = f'<div class="border">{note}</div>' if note else ""
    return (f'<div class="rcard {kind}"><div class="rhead"><span class="ico">{icon}</span>'
            f'<div><div class="rtitle">{title}</div><div class="rsub">Risk score</div></div>'
            f'<span class="pill" style="--c:{COLORS[lvl]}">{lvl}</span></div>'
            f'<div class="ring" style="--p:{pct};--c:{COLORS[lvl]}"><div class="ringtxt"><b>{pct}%</b><span>{lvl} risk</span></div></div>'
            f'<div class="scale"><span style="left:{min(max(pct, 2), 98)}%"></span></div>'
            f'<div class="scalel"><small>Low</small><small>Moderate</small><small>High</small></div>'
            f'<div class="rtext">{MEANING[lvl]}</div>{note_html}</div>')


def tips_card(title, icon, p, base_tips, extra_tips):
    lvl = level(p)
    col = COLORS[lvl]
    items = "".join(f'<div class="titem extra" style="--c:{col}"><span class="te">{e}</span><span>{t}</span></div>'
                    for e, t in extra_tips)
    items += "".join(f'<div class="titem"><span class="te">{e}</span><span>{t}</span></div>' for e, t in base_tips)
    return (f'<div class="tcard" style="--c:{col}"><div class="thead"><span>{icon}</span><b>{title}</b>'
            f'<span class="pill" style="--c:{col}">{lvl} risk</span></div>{items}</div>')


def food_html(disease, vegetarian):
    out = ""
    for group, items in FOOD[disease].items():
        lines = []
        for text, veg in items:
            if vegetarian:
                if veg is None:
                    continue
                text = text if veg == SAME else veg
            lines.append(f"<li>{text}</li>")
        out += f'<div class="fgroup" style="--c:{FOOD_COLORS[group]}"><h5>{group}</h5><ul>{"".join(lines)}</ul></div>'
    return out


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


# =====================================================================  PAGE  =====
st.markdown(
    '<div class="hero"><span class="tag"><i></i>Free health risk check</span>'
    '<h1>Check your health risk in 2 minutes</h1>'
    '<p>Answer a few simple questions to get an easy-to-read estimate of your diabetes and heart disease risk.</p>'
    '<svg class="ecg" viewBox="0 0 420 44" preserveAspectRatio="none"><path d="M0 24 H70 L82 24 L92 8 L104 40 '
    'L114 24 H190 L202 24 L212 6 L224 42 L234 24 H310 L322 24 L332 10 L344 38 L354 24 H420"/></svg></div>'
    '<div class="chips"><span class="chip c4">🔒 Your answers are not saved</span><span class="chip c2">🩸 Diabetes</span>'
    '<span class="chip c3">❤️ Heart disease</span></div>'
    '<div class="steps"><div class="stp"><b>1</b>Enter your details</div><div class="stp"><b>2</b>Tap Check my risk</div>'
    '<div class="stp"><b>3</b>Read your results</div></div>',
    unsafe_allow_html=True,
)

if "quote" not in st.session_state:
    st.session_state["quote"] = random.choice(QUOTES)
st.markdown(quote_html(*st.session_state["quote"]), unsafe_allow_html=True)
st.write("")

# ---------------- About you ----------------
with st.container(border=True):
    st.markdown(section_header("👤", "About you", "Basic details and body measurements"), unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    age = c1.number_input("Age (years)", 1, 100, 40)
    sex = c2.selectbox("Sex", ["Male", "Female"])
    h_col, w_col = st.columns(2)
    height = h_col.number_input("Height (cm)", 100.0, 250.0, 170.0, step=0.5)
    weight = w_col.number_input("Weight (kg)", 20.0, 300.0, 70.0, step=0.5)
    bmi_raw = weight / ((height / 100) ** 2)
    bmi = round(min(max(bmi_raw, 10.0), 70.0), 1)   # keep inside the range the model was trained on
    st.markdown(bmi_card(bmi_raw), unsafe_allow_html=True)
    if bmi_raw != bmi:
        st.warning("This BMI is outside the usual range (10 to 70). Please re-check your height and weight.")
    preg = 0
    if sex == "Female":
        preg = st.number_input("Number of pregnancies", 0, 20, 0, help="Enter 0 if none.")

# ---------------- Health numbers ----------------
with st.container(border=True):
    st.markdown(section_header("🩺", "Your health numbers", "Use your latest check-up values. Not sure? Keep the default."),
                unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    sbp = b1.number_input("Blood pressure: top number (mmHg)", 80, 220, 120,
                          help="Systolic pressure, for example the 120 in 120/80.")
    dbp = b2.number_input("Blood pressure: bottom number (mmHg)", 30, 140, 80,
                          help="Diastolic pressure, for example the 80 in 120/80.")
    g1, g2 = st.columns(2)
    glucose = g1.number_input("Fasting blood sugar (mg/dL)", 40, 400, 95,
                              help="Blood sugar after 8+ hours without food. Normal is under 100 mg/dL.")
    chol = g2.number_input("Total cholesterol (mg/dL)", 100, 600, 190,
                           help="From a lipid profile blood test. Healthy is under 200 mg/dL.")
    cp_label = st.selectbox("Do you have chest pain or discomfort?", list(CHEST_PAIN), index=0)
    cp = CHEST_PAIN[cp_label]
    e1, e2 = st.columns(2)
    exang = e1.selectbox("Chest pain or breathlessness during exercise?", ["No", "Yes"])
    family = e2.selectbox("Family history of diabetes", list(FAMILY))
    dpf = FAMILY[family]

# ---------------- Optional lab / ECG report ----------------
HR_DEFAULT = max(60, min(220, int(0.9 * (220 - age))))
skin, insulin = 25, 80
ecg, hr, oldpeak, slope, ca, thal = 0, HR_DEFAULT, 0.0, 1, 0, 3
with st.expander("📄 Optional: I have lab or ECG reports"):
    st.markdown('<div class="hint">These come from lab or hospital reports. If you do not have them, leave the typical '
                'healthy values already filled in. Your result will be a little less precise.</div>', unsafe_allow_html=True)
    st.markdown('<div class="subhead">🩸 Diabetes tests</div>', unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    insulin = d1.number_input("Insulin level (mu U/ml)", 0, 900, 80)
    skin = d2.number_input("Skin fold thickness (mm)", 5, 100, 25)
    st.markdown('<div class="subhead">❤️ Heart tests</div>', unsafe_allow_html=True)
    h1, h2 = st.columns(2)
    ecg = h1.selectbox("Resting ECG result", list(ECG_LABELS), format_func=lambda k: ECG_LABELS[k])
    hr = h2.number_input("Highest heart rate in exercise test (bpm)", 60, 220, HR_DEFAULT)
    slope = h1.selectbox("Exercise ECG slope", list(SLOPE_LABELS), format_func=lambda k: SLOPE_LABELS[k])
    oldpeak = h2.number_input("ST depression (oldpeak)", 0.0, 7.0, 0.0, step=0.1,
                              help="Drop in the ECG trace during exercise, from the report. 0 is typical for a healthy heart.")
    ca = h1.selectbox("Blocked vessels seen on scan (0 to 3)", [0, 1, 2, 3])
    thal = h2.selectbox("Thalassemia blood test", list(THAL_LABELS), format_func=lambda k: THAL_LABELS[k])

used_typical = (insulin == 80 and skin == 25 and ecg == 0 and hr == HR_DEFAULT and slope == 1
                and oldpeak == 0.0 and ca == 0 and thal == 3)

IST = timezone(timedelta(hours=5, minutes=30))
inputs_sig = (age, sex, height, weight, preg, sbp, dbp, glucose, chol, cp_label, exang, family,
              skin, insulin, ecg, hr, oldpeak, slope, ca, thal)

st.write("")
if st.button("🔍 Check my risk", type="primary", use_container_width=True):
    d_row = pd.DataFrame([[preg, glucose, dbp, skin, insulin, bmi, dpf, age]],
                         columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                                  "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"])
    p_d = models["diabetes"].predict_proba(d_row)[0, 1]

    h_cols = ["age", "sex", "chest_pain_type", "resting_bp", "cholesterol",
              "fasting_blood_sugar", "resting_ecg", "max_heart_rate", "exercise_angina",
              "oldpeak", "slope", "num_major_vessels", "thalassemia"]

    def heart_prob(fbs_value):
        row = pd.DataFrame([[age, int(sex == "Male"), cp, sbp, chol, fbs_value, ecg, hr,
                             int(exang == "Yes"), oldpeak, slope, ca, thal]], columns=h_cols)
        return models["heart"].predict_proba(row)[0, 1]

    snap = {"time": datetime.now(IST), "sig": inputs_sig, "age": age, "sex": sex, "bmi": bmi,
            "glucose": glucose, "sbp": sbp, "chol": chol, "used_typical": used_typical,
            "p_d": p_d, "p_h": heart_prob(int(glucose > 120)), "p_lo": heart_prob(0), "p_hi": heart_prob(1)}
    history = st.session_state.setdefault("history", [])   # kept only while this page stays open
    history.append(snap)
    del history[:-5]
    st.session_state["last"] = snap

snap = st.session_state.get("last")
if snap:
    # results always come from the moment the button was pressed
    age, sex, bmi = snap["age"], snap["sex"], snap["bmi"]
    glucose, sbp, chol = snap["glucose"], snap["sbp"], snap["chol"]
    p_d, p_h, p_lo, p_hi = snap["p_d"], snap["p_h"], snap["p_lo"], snap["p_hi"]
    used_typical = snap["used_typical"]
    stamp = snap["time"].strftime("%d %b %Y, %I:%M %p IST")

    # ---------------- results ----------------
    st.markdown('<div class="section-title">Your results</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="stamp">🕒 Last checked: {stamp}</div>', unsafe_allow_html=True)
    if snap["sig"] != inputs_sig:
        st.info("You changed some answers after this check. Tap **Check my risk** again to update the results.")
    st.markdown('<div class="legend"><span style="--c:#16A34A">Low: under 33%</span>'
                '<span style="--c:#D97706">Moderate: 33% to 65%</span><span style="--c:#DC2626">High: 66% or more</span></div>'
                '<p class="scorenote">These are risk scores from a computer model, not the exact chance of getting a disease.</p>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="rgrid">{result_card("Diabetes", p_d, "d", "🩸")}'
                f'{result_card("Heart disease", p_h, "h", "❤️")}</div>', unsafe_allow_html=True)
    if used_typical:
        st.caption("The optional lab and ECG details used typical healthy values, so the heart result is less precise.")
    if len(st.session_state.get("history", [])) >= 2:
        prev = st.session_state["history"][-2]
        pstamp = prev["time"].strftime("%d %b %Y, %I:%M %p")
        dd, dh = round((p_d - prev["p_d"]) * 100), round((p_h - prev["p_h"]) * 100)
        st.markdown(f'<div class="cmp"><b>Compared with your previous check ({pstamp}):</b><br>'
                    f'🩸 Diabetes: {round(prev["p_d"] * 100)}% → {round(p_d * 100)}% ({dd:+d} points)<br>'
                    f'❤️ Heart disease: {round(prev["p_h"] * 100)}% → {round(p_h * 100)}% ({dh:+d} points)<br>'
                    '<small>Checks are kept only while this page stays open. Nothing is saved.</small></div>',
                    unsafe_allow_html=True)

    # ---------------- tips ----------------
    bmi_cat, _ = bmi_category(bmi)
    d_extra, h_extra = [], []
    if bmi >= 25:
        d_extra.append(("⚖️", f"Your BMI is {bmi:.1f} ({bmi_cat}). Gradual weight loss can lower your diabetes risk."))
    elif bmi < 18.5:
        d_extra.append(("⚖️", f"Your BMI is {bmi:.1f} (Underweight). Aim for balanced, nutritious meals."))
    if glucose >= 100:
        d_extra.append(("🩸", f"Your fasting glucose is {glucose} mg/dL, above the normal level (under 100). Discuss it with a doctor."))
    if sbp >= 130:
        h_extra.append(("💓", f"Your top blood pressure is {sbp} mmHg, above the healthy level (under 130). Reduce salt and get it rechecked."))
    if chol >= 240:
        h_extra.append(("🧈", f"Your cholesterol is {chol} mg/dL, which is high. Cut saturated fat and ask about a lipid profile."))
    if bmi >= 25:
        h_extra.append(("⚖️", f"A BMI of {bmi:.1f} adds strain on the heart. Weight management helps."))

    st.markdown('<div class="section-title">Health tips for you</div>', unsafe_allow_html=True)
    st.markdown('<div class="alert"><b>Get urgent medical help</b> if you have chest pain, sudden breathlessness, '
                'fainting, or pain spreading to your arm or jaw, whatever your result says.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="tgrid">'
        f'{tips_card("Diabetes tips", "🩸", p_d, TIPS["diabetes"][level(p_d)], d_extra)}'
        f'{tips_card("Heart health tips", "❤️", p_h, TIPS["heart"][level(p_h)], h_extra)}'
        '</div>', unsafe_allow_html=True)

    # ---------------- food guide ----------------
    st.markdown('<div class="section-title">Your food guide</div>', unsafe_allow_html=True)
    veg = st.toggle("🌱 Show vegetarian options only", value=False)
    st.markdown('<div class="plate"><div class="pie"></div><div><b>The healthy plate</b><ul>'
                '<li style="--c:#16A34A">Half: vegetables and salad</li>'
                '<li style="--c:#D97706">One quarter: whole grains, millets or roti</li>'
                '<li style="--c:#2563EB">One quarter: dal' + ('' if veg else ', egg, fish or lean meat') + ', paneer or tofu</li>'
                '</ul></div></div>', unsafe_allow_html=True)
    order = ["diabetes", "heart"] if p_d >= p_h else ["heart", "diabetes"]
    titles = {"diabetes": "🩸 Diabetes-friendly eating", "heart": "❤️ Heart-healthy eating"}
    for i, dz in enumerate(order):
        with st.expander(titles[dz] + ("  (higher risk, start here)" if i == 0 else ""), expanded=(i == 0)):
            st.markdown(food_html(dz, veg), unsafe_allow_html=True)
    st.markdown('<div class="fnote">Eat regular meals, watch portion sizes, drink plenty of water and stay active. '
                'If you are pregnant, have kidney disease or take medicines for diabetes or blood pressure, '
                "follow your doctor's or dietitian's advice.</div>", unsafe_allow_html=True)

    # ---------------- blood sugar effect ----------------
    st.markdown('<div class="section-title">Does blood sugar affect your heart risk?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel">'
        f'<p>We compared the same person with normal and high fasting blood sugar (above 120 mg/dL). '
        f'For this profile, high blood sugar changes the heart risk score by about <b>{(p_hi - p_lo) * 100:+.0f} points</b>.</p>'
        f'{bar("Normal blood sugar", p_lo)}{bar("High blood sugar", p_hi)}</div>',
        unsafe_allow_html=True,
    )

    with st.expander("🔎 Curious how the model decides?"):
        st.markdown(f'<div class="fgrid">{factors_card("Diabetes model", "d", models["diabetes"])}'
                    f'{factors_card("Heart model", "h", models["heart"])}</div>', unsafe_allow_html=True)
        st.caption("Importance is measured across the whole training data, not for this one person.")

    # ---------------- download ----------------
    lines = ["DISEASE RISK PREDICTION - MY SUMMARY", f"Checked on: {stamp}", "",
             f"Age: {age} | Sex: {sex} | BMI: {bmi:.1f} ({bmi_cat})", "",
             f"Diabetes risk score: {round(p_d * 100)}% ({level(p_d)})",
             f"Heart disease risk score: {round(p_h * 100)}% ({level(p_h)})", "", "Diabetes tips:"]
    lines += [f"- {t}" for _, t in d_extra + TIPS["diabetes"][level(p_d)]]
    lines += ["", "Heart health tips:"] + [f"- {t}" for _, t in h_extra + TIPS["heart"][level(p_h)]]
    lines += ["", "This is an educational risk score, not a medical diagnosis. Please consult a doctor."]
    st.write("")
    st.download_button("⬇️ Download my summary", "\n".join(lines), file_name="my_risk_summary.txt",
                       use_container_width=True)

    st.markdown(
        '<div class="note">This is an educational risk score from a Random Forest model, not a diagnosis and not the '
        'exact chance of getting a disease. The tips are general health guidance, not medical treatment. The diabetes '
        'and heart models are trained on separate datasets, so the blood sugar link is an association the model '
        'learned, not proof of cause. Please consult a doctor for medical advice.</div>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="foot">Disease Risk Prediction · Random Forest · Streamlit<br>'
            'Team:45 - Artificial Intelligence and Data Science · HNN College of Engineering</div>', unsafe_allow_html=True)
