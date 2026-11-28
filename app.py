import random
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline

st.set_page_config(page_title="Disease Risk Prediction", page_icon="🩺", layout="wide")

D_FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
              "DiabetesPedigreeFunction", "Age"]
H_FEATURES = ["age", "sex", "chest_pain_type", "resting_bp", "cholesterol", "fasting_blood_sugar", "resting_ecg",
              "max_heart_rate", "exercise_angina", "oldpeak", "slope", "num_major_vessels", "thalassemia"]
ZERO_MEANS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

COLORS = {"Low": "#16A34A", "Moderate": "#D97706", "High": "#DC2626"}
ORDER = {"Low": 0, "Moderate": 1, "High": 2}
QUOTES = [
    ("Prevention is better than cure.", "Proverb"),
    ("Health is wealth.", "Proverb"),
    ("An apple a day keeps the doctor away.", "Proverb"),
    ("The greatest wealth is health.", "Virgil"),
    ("Early to bed and early to rise makes a man healthy, wealthy, and wise.", "Benjamin Franklin"),
]
RESULT_QUOTES = {"Low": QUOTES[1], "Moderate": QUOTES[0], "High": QUOTES[0]}
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

# Words shown to the user -> codes used by the models.
CHEST_PAIN = {"None": 4, "Low": 3, "Moderate": 2, "Severe": 1}
CHEST_PAIN_TYPE = {"None": "asymptomatic (no chest pain)", "Low": "non-anginal pain",
                   "Moderate": "atypical angina", "Severe": "typical angina"}
ECG = {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}
SLOPE = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}
THAL = {"Normal": 3, "Fixed defect": 6, "Reversible defect": 7}
UNKNOWN = "I don't know"

FRIENDLY = {
    "Pregnancies": "Pregnancies", "Glucose": "Blood glucose (2-hour test)", "BloodPressure": "Blood pressure (bottom)",
    "SkinThickness": "Skin fold thickness", "Insulin": "Insulin level", "BMI": "BMI",
    "DiabetesPedigreeFunction": "Family history score", "Age": "Age", "age": "Age", "sex": "Sex",
    "chest_pain_type": "Chest pain", "resting_bp": "Blood pressure (top)", "cholesterol": "Cholesterol",
    "fasting_blood_sugar": "Fasting blood sugar", "resting_ecg": "Resting ECG", "max_heart_rate": "Max heart rate",
    "exercise_angina": "Exercise chest pain", "oldpeak": "ST depression", "slope": "ST slope",
    "num_major_vessels": "Major vessels", "thalassemia": "Thalassemia",
}

# Share of genes held in common with the relative (as used in the published diabetes pedigree function).
RELATIVES = {
    "Mother": 0.5, "Father": 0.5, "Full brother or sister": 0.5,
    "Half brother or sister": 0.25, "Grandparent": 0.25, "Aunt or uncle": 0.25,
    "Half aunt or half uncle": 0.125, "First cousin": 0.125,
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
.stTabs [data-baseweb="tab-panel"] {margin-top: .8rem; padding: 1rem 1.2rem .6rem; border-radius: 18px;
  border: 1px solid rgba(37,99,235,.25); background: rgba(37,99,235,.05);}
.bmi {display: inline-block; margin: .1rem 0 .4rem; padding: .35rem .8rem; border-radius: 99px; font-weight: 700; font-size: .88rem;
  background: rgba(37,99,235,.14); border: 1px solid rgba(37,99,235,.35);}

  background-image: linear-gradient(100deg, #1D4ED8, #0D9488); background-color: #1D4ED8; color: #fff;
  box-shadow: 0 14px 28px -14px rgba(29,78,216,.9);}
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

.stTabs [data-baseweb="tab"] {font-weight: 700; font-size: 1rem;}
div[data-testid="stButton"], div[data-testid="stDownloadButton"] {width: 100%;}
div[data-testid="stButton"] button, div[data-testid="stDownloadButton"] button {
  width: 100%; border-radius: 12px; font-weight: 800; padding: .8rem 1rem; font-size: 1.06rem; transition: transform .15s ease, box-shadow .15s ease;}
div[data-testid="stButton"] button[kind="primary"], div[data-testid="stButton"] button[data-testid="stBaseButton-primary"] {
  background-image: linear-gradient(100deg, #1D4ED8, #0D9488); background-color: #1D4ED8; color: #fff; border: none;
  box-shadow: 0 14px 28px -14px rgba(29,78,216,.9);}
div[data-testid="stButton"] button[kind="primary"]:hover, div[data-testid="stButton"] button[data-testid="stBaseButton-primary"]:hover {
  transform: translateY(-2px); color: #fff; box-shadow: 0 18px 32px -14px rgba(13,148,136,.9);}
.rangetxt {text-align: center; font-size: .85rem; font-weight: 700; opacity: .85; margin-top: .2rem;}
.rangebar {position: relative; height: 10px; border-radius: 99px; background: rgba(128,128,128,.25); margin: .5rem auto 1.1rem; max-width: 260px;}
.rangebar .seg {position: absolute; top: 0; bottom: 0; border-radius: 99px; background: var(--c); opacity: .6;}
.rangebar .mark {position: absolute; top: -4px; width: 4px; height: 18px; border-radius: 2px; background: #fff; transform: translateX(-50%); box-shadow: 0 0 0 1px rgba(0,0,0,.55);}
.urgent {margin-top: 1rem; padding: 1rem 1.2rem; border-radius: 16px; border: 1px solid rgba(220,38,38,.5); background: rgba(220,38,38,.10); font-size: .92rem; line-height: 1.55;}
.urgent b {display: block; margin-bottom: .2rem;}
.alert {margin-top: .8rem; padding: .7rem 1rem; border-radius: 12px; font-size: .9rem; line-height: 1.5; font-weight: 700; border: 1px solid rgba(220,38,38,.5); background: rgba(220,38,38,.12);}
.caution {margin-top: .8rem; padding: .7rem 1rem; border-radius: 12px; font-size: .88rem; line-height: 1.5; border: 1px dashed rgba(37,99,235,.6); background: rgba(37,99,235,.10);}
.mgrid {display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: .8rem; margin: .4rem 0 1rem;}
.mtile {padding: 1rem 1rem; border-radius: 16px; border: 1px solid rgba(128,128,128,.25); background: rgba(128,128,128,.07); text-align: center;}
.mtile b {display: block; font-size: 1.6rem; font-weight: 800; line-height: 1.1; background: linear-gradient(100deg, #2563EB, #0D9488); -webkit-background-clip: text; background-clip: text; color: transparent;}
.mtile span {font-size: .8rem; opacity: .75; font-weight: 600;}
.cm {border-collapse: collapse; margin: .4rem 0 1rem; font-size: .92rem;}
.cm th, .cm td {padding: .6rem 1rem; border: 1px solid rgba(128,128,128,.3); text-align: center;}
.cm th {font-size: .8rem; opacity: .75;}
.cm td b {font-size: 1.2rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ models (trained when the app starts)
def build(X, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    def new_model(trees=200, seed=42):
        return make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=trees, random_state=seed))

    model = new_model().fit(Xtr, ytr)
    pred, prob = model.predict(Xte), model.predict_proba(Xte)[:, 1]
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    cv = cross_val_score(new_model(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="accuracy").mean()
    rng = np.random.default_rng(42)  # 20 smaller forests on resampled data, used for the likely score range
    boots = []
    for i in range(20):
        idx = rng.integers(0, len(Xtr), len(Xtr))
        boots.append(new_model(60, i).fit(Xtr.iloc[idx], ytr.iloc[idx]))
    return {
        "model": model, "boots": boots, "features": list(X.columns),
        "medians": pd.Series(model[0].statistics_, index=X.columns),
        "importance": pd.Series(model[-1].feature_importances_, index=X.columns).sort_values(ascending=False),
        "metrics": {"Accuracy": accuracy_score(yte, pred), "Precision": precision_score(yte, pred),
                    "Recall": recall_score(yte, pred), "F1 score": f1_score(yte, pred),
                    "ROC-AUC": roc_auc_score(yte, prob), "5-fold CV accuracy": cv},
        "cm": (tn, fp, fn, tp), "n": len(X), "n_test": len(yte), "pos_rate": float(y.mean()),
    }


@st.cache_resource(show_spinner="Training the Random Forest models...")
def load_models():
    base, out = Path(__file__).parent, {}
    for name, fname, feats, target in (("diabetes", "diabetes.csv", D_FEATURES, "Outcome"),
                                       ("heart", "heart.csv", H_FEATURES, "target")):
        path = base / fname
        if not path.exists():
            path = Path(fname)
        if not path.exists():
            return "missing:" + fname
        df = pd.read_csv(path)
        if df.shape[1] != len(feats) + 1:
            return "columns:" + fname
        df.columns = feats + [target]  # assigned by position
        df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=[target])
        if name == "diabetes":
            for c in ZERO_MEANS_MISSING:
                df[c] = df[c].replace(0, np.nan)  # 0 is not a real reading, treat as missing
            y = df[target].astype(int)
        else:
            vals = set(df[target].unique())
            y = (df[target] == 2).astype(int) if vals == {1, 2} else (df[target] > 0).astype(int)
        out[name] = build(df[feats], y)
    return out


M = load_models()
if isinstance(M, str):
    kind, fname = M.split(":")
    msg = (f"Add your <b>{fname}</b> file to the same folder as app.py in your GitHub repository, then reload this page."
           if kind == "missing" else f"{fname} does not have the expected number of columns. Check the column order.")
    st.markdown(f'<div class="notice"><b>Dataset problem</b>{msg}</div>', unsafe_allow_html=True)
    st.stop()
DM, HM = M["diabetes"], M["heart"]


# ------------------------------------------------------------------ helpers
def level(p):
    return "Low" if p < 0.33 else "Moderate" if p < 0.66 else "High"


def quote_html(text, author):
    return f'<div class="quote">&ldquo;{text}&rdquo;<small>{author}</small></div>'


def result_card(title, p, lo, hi, kind, icon, key):
    score, lvl = round(p * 100), level(p)
    return (f'<div class="rcard {kind}"><div class="rhead"><span class="ico">{icon}</span>'
            f'<div><div class="rtitle">{title}</div><div class="rsub">Model score</div></div>'
            f'<span class="pill" style="--c:{COLORS[lvl]}">{lvl}</span></div>'
            f'<div class="ring" style="--p:{score};--c:{COLORS[lvl]}"><div class="ringtxt"><b>{score}</b><span>out of 100</span></div></div>'
            f'<div class="lvl" style="--c:{COLORS[lvl]}">{lvl} risk category</div>'
            f'<div class="rangetxt">Likely range: {round(lo * 100)} to {round(hi * 100)}</div>'
            f'<div class="rangebar" style="--c:{COLORS[lvl]}"><div class="seg" style="left:{lo * 100:.0f}%;width:{max((hi - lo) * 100, 2):.0f}%"></div>'
            f'<div class="mark" style="left:{score}%"></div></div>'
            f'<div class="rtext">{MEANING[lvl]}</div>'
            f'<div class="tip"><b>What you can do</b>{TIPS[key][lvl]}</div></div>')


def bar(label, p):
    pct = round(p * 100)
    return (f'<div class="barrow"><span class="barl">{label}</span>'
            f'<div class="track"><div class="fill" style="width:{pct}%;background:{COLORS[level(p)]}"></div></div>'
            f'<span class="barv">{pct}</span></div>')


def predict(entry, vals):
    return float(entry["model"].predict_proba(pd.DataFrame([vals], columns=entry["features"]))[0, 1])


def predict_range(entry, vals, p):
    df = pd.DataFrame([vals], columns=entry["features"])
    scores = [float(m.predict_proba(df)[0, 1]) for m in entry["boots"]]
    lo, hi = np.percentile(scores, [10, 90])
    return min(float(lo), p), max(float(hi), p)


def drivers(entry, vals, base, top=4):
    """Swap each of your values for the typical (median) value and see how the model score changes."""
    out = []
    for k in entry["features"]:
        if k == "sex" or pd.isna(vals[k]):
            continue
        alt = dict(vals)
        alt[k] = float(entry["medians"][k])
        out.append((k, base - predict(entry, alt)))
    out = [x for x in sorted(out, key=lambda x: -abs(x[1])) if abs(x[1]) >= 0.005]
    return out[:top]


def drivers_card(title, kind, items):
    if not items:
        body = '<div class="rsub">No single value stood out compared with a typical person in the data.</div>'
    else:
        mx = max(abs(d) for _, d in items) or 1
        body = "".join(
            f'<div class="barrow"><span class="barl">{"▲" if d > 0 else "▼"} {FRIENDLY[k]}</span>'
            f'<div class="track"><div class="fill" style="width:{abs(d) / mx * 100:.0f}%;background:{"#DC2626" if d > 0 else "#16A34A"}"></div></div>'
            f'<span class="barv">{d * 100:+.0f}</span></div>' for k, d in items)
    return (f'<div class="fcard {kind}"><h4>{title}</h4>'
            f'<small>▲ raises your score · ▼ lowers it (points out of 100), compared with a typical person in the data</small>{body}</div>')


def importance_card(title, kind, entry, top=5):
    imp = entry["importance"].head(top)
    mx = imp.max() or 1
    rows = "".join(
        f'<div class="barrow"><span class="barl">{FRIENDLY[n]}</span>'
        f'<div class="track"><div class="fill" style="width:{v / mx * 100:.0f}%"></div></div>'
        f'<span class="barv">{v * 100:.0f}%</span></div>' for n, v in imp.items())
    return f'<div class="fcard static {kind}"><h4>{title}</h4><small>Most relied-on inputs overall</small>{rows}</div>'


def plate_html():
    return ('<div class="plate"><div class="disc"></div><ul><h4>The healthy plate</h4>'
            '<li><i style="background:#16A34A"></i>Half: vegetables and salad</li>'
            '<li><i style="background:#F59E0B"></i>One quarter: whole grains, millets or roti</li>'
            '<li><i style="background:#2563EB"></i>One quarter: dal, egg, fish or lean meat</li></ul></div>')


def bmi_category(v):
    return "Underweight" if v < 18.5 else "Normal" if v < 25 else "Overweight" if v < 30 else "Obese range"


def compute_dpf(rels):
    """Diabetes pedigree function (Smith et al., 1988): relatives with diabetes raise the score, relatives without lower it."""
    num = 20 + sum(RELATIVES[r["rel"]] * max(88 - r["age"], 0) for r in rels if r["dm"])
    den = 50 + sum(RELATIVES[r["rel"]] * max(r["age"] - 14, 0) for r in rels if not r["dm"])
    return round(num / den, 3)


def remove_last_relative():
    if st.session_state.get("relatives"):
        st.session_state["relatives"].pop()


def clear_relatives():
    st.session_state["relatives"] = []


def with_unknown(col, label, mn, mx, key, help=None):
    """Number input plus an 'I don't know' checkbox. Unknown values become missing and are filled with the training median."""
    box = col.container()
    unknown = col.checkbox("I don't know", key=f"{key}_unk")
    val = box.number_input(label, mn, mx, key=key, help=help, disabled=unknown)
    return (np.nan, True) if unknown else (val, False)


def metrics_block(title, entry):
    m = entry["metrics"]
    tn, fp, fn, tp = entry["cm"]
    st.markdown(f'<div class="sec">{title}</div>', unsafe_allow_html=True)
    st.markdown('<div class="mgrid">' + "".join(
        f'<div class="mtile"><b>{v * 100:.0f}%</b><span>{k}</span></div>' for k, v in m.items()) + '</div>',
        unsafe_allow_html=True)
    st.markdown(
        '<table class="cm"><tr><th></th><th>Predicted: no</th><th>Predicted: yes</th></tr>'
        f'<tr><th>Actual: no</th><td><b>{tn}</b></td><td><b>{fp}</b></td></tr>'
        f'<tr><th>Actual: yes</th><td><b>{fn}</b></td><td><b>{tp}</b></td></tr></table>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="help"><b>Recall</b>: {tp} of {tp + fn} real cases caught. <b>Precision</b>: {tp} of {tp + fp} flagged '
        f'people really had it. Based on {entry["n"]} records ({entry["pos_rate"] * 100:.0f}% positive), with {entry["n_test"]} '
        'held out for testing.</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ state, callbacks
DEFAULTS = {
    "age": 35, "sex": "Female", "preg": 0, "height": 160, "weight": 60,
    "sbp": 120, "sbp_unk": False, "dbp": int(round(DM["medians"]["BloodPressure"])), "dbp_unk": False,
    "chol": int(round(HM["medians"]["cholesterol"])), "chol_unk": True,
    "fbsv": 95, "fbsv_unk": True,
    "ogtt": "No or not sure", "glucose": int(round(DM["medians"]["Glucose"])),
    "cp": "None", "exang": "No",
    "insulin": int(round(DM["medians"]["Insulin"])), "insulin_unk": True,
    "skin": int(round(DM["medians"]["SkinThickness"])), "skin_unk": True,
    "maxhr": int(round(HM["medians"]["max_heart_rate"])), "maxhr_unk": True,
    "oldpeak": 1.0, "oldpeak_unk": True,
    "ecg": UNKNOWN, "slope": UNKNOWN, "ca": UNKNOWN, "thal": UNKNOWN,
}
EXAMPLES = {
    "low": {"inputs": {"age": 28, "sex": "Female", "height": 165, "weight": 58, "sbp": 110, "dbp": 68, "chol": 170,
                       "chol_unk": False, "fbsv": 88, "fbsv_unk": False, "ogtt": "Yes", "glucose": 92,
                       "cp": "None", "exang": "No"},
            "relatives": [{"rel": "Mother", "dm": False, "age": 54}, {"rel": "Father", "dm": False, "age": 58}]},
    "high": {"inputs": {"age": 56, "sex": "Female", "preg": 4, "height": 158, "weight": 84, "sbp": 152, "dbp": 92,
                        "chol": 262, "chol_unk": False, "fbsv": 148, "fbsv_unk": False, "ogtt": "Yes", "glucose": 178,
                        "cp": "Severe", "exang": "Yes", "insulin": 210, "insulin_unk": False,
                        "skin": 40, "skin_unk": False},
             "relatives": [{"rel": "Mother", "dm": True, "age": 45}, {"rel": "Father", "dm": True, "age": 52},
                           {"rel": "Full brother or sister", "dm": True, "age": 40}]},
}
for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)
st.session_state.setdefault("view", "form")
st.session_state.setdefault("relatives", [])


def show_results():
    st.session_state["saved"] = {k: st.session_state.get(k, v) for k, v in DEFAULTS.items()}
    st.session_state["view"] = "results"
    st.session_state["scroll_top"] = True


def edit_answers():
    for k, v in st.session_state.get("saved", {}).items():
        st.session_state[k] = v
    st.session_state["view"] = "form"


def load_example(name):
    for k, v in EXAMPLES[name]["inputs"].items():
        st.session_state[k] = v
    st.session_state["relatives"] = [dict(r) for r in EXAMPLES[name]["relatives"]]


def reset_form():
    st.session_state.clear()


BRAND = ('<div class="brand"><span class="logo">🩺</span><div><b>Disease Risk Prediction</b>'
         '<small>Diabetes and heart screening</small></div><span class="bt">AI-powered screening</span></div>')

if st.session_state.get("view") == "results" and "saved" in st.session_state:
    # ============================================================== RESULTS VIEW
    st.markdown(BRAND, unsafe_allow_html=True)
    if st.session_state.pop("scroll_top", False):
        components.html(
            "<script>const d = window.parent.document;"
            "const el = d.querySelector('[data-testid=\"stMain\"]') || d.querySelector('section.main') || d.documentElement;"
            "try { el.scrollTo({top: 0}); window.parent.scrollTo({top: 0}); } catch (e) {}</script>", height=0)
    st.button("← Edit my answers", on_click=edit_answers, key="edit_top")

    S = st.session_state["saved"]
    rels = st.session_state.get("relatives", [])
    nan = np.nan
    age, sex = S["age"], S["sex"]
    preg = S["preg"] if sex == "Female" else 0
    bmi = round(S["weight"] / ((S["height"] / 100) ** 2), 1)
    sbp = nan if S["sbp_unk"] else S["sbp"]
    dbp = nan if S["dbp_unk"] else S["dbp"]
    chol = nan if S["chol_unk"] else S["chol"]
    fasting = nan if S["fbsv_unk"] else S["fbsv"]
    glucose = S["glucose"] if S["ogtt"] == "Yes" else nan
    insulin = nan if S["insulin_unk"] else S["insulin"]
    skin = nan if S["skin_unk"] else S["skin"]
    maxhr = nan if S["maxhr_unk"] else S["maxhr"]
    oldpeak = nan if S["oldpeak_unk"] else S["oldpeak"]
    ecg = nan if S["ecg"] == UNKNOWN else ECG[S["ecg"]]
    slope = nan if S["slope"] == UNKNOWN else SLOPE[S["slope"]]
    ca = nan if S["ca"] == UNKNOWN else int(S["ca"])
    thal = nan if S["thal"] == UNKNOWN else THAL[S["thal"]]
    dpf = compute_dpf(rels) if rels else nan
    fbs_flag = nan if pd.isna(fasting) else int(fasting > 120)

    assumed = [name for cond, name in (
        (S["ogtt"] != "Yes", "blood glucose from a glucose tolerance test"), (S["sbp_unk"], "blood pressure (top number)"),
        (S["dbp_unk"], "blood pressure (bottom number)"), (S["chol_unk"], "cholesterol"),
        (S["fbsv_unk"], "fasting blood sugar"), (S["insulin_unk"], "insulin level"),
        (S["skin_unk"], "skin fold thickness"), (not rels, "family history score"),
        (S["maxhr_unk"], "highest exercise heart rate"), (S["oldpeak_unk"], "ST depression"),
        (S["ecg"] == UNKNOWN, "resting ECG"), (S["slope"] == UNKNOWN, "exercise ECG slope"),
        (S["ca"] == UNKNOWN, "blocked vessels on scan"), (S["thal"] == UNKNOWN, "thalassemia test")) if cond]

    d_vals = {"Pregnancies": float(preg), "Glucose": glucose, "BloodPressure": dbp, "SkinThickness": skin,
              "Insulin": insulin, "BMI": bmi, "DiabetesPedigreeFunction": dpf, "Age": float(age)}
    h_vals = {"age": float(age), "sex": int(sex == "Male"), "chest_pain_type": CHEST_PAIN[S["cp"]], "resting_bp": sbp,
              "cholesterol": chol, "fasting_blood_sugar": fbs_flag, "resting_ecg": ecg, "max_heart_rate": maxhr,
              "exercise_angina": int(S["exang"] == "Yes"), "oldpeak": oldpeak, "slope": slope,
              "num_major_vessels": ca, "thalassemia": thal}
    with st.spinner("Analyzing your inputs..."):
        p_d, p_h = predict(DM, d_vals), predict(HM, h_vals)
        lo_d, hi_d = predict_range(DM, d_vals, p_d)
        lo_h, hi_h = predict_range(HM, h_vals, p_h)
        drv_d, drv_h = drivers(DM, d_vals, p_d), drivers(HM, h_vals, p_h)
        p_lo = predict(HM, {**h_vals, "fasting_blood_sugar": 0})
        p_hi = predict(HM, {**h_vals, "fasting_blood_sugar": 1})
        whatif = predict(DM, {**d_vals, "BMI": 24.9}) if bmi > 24.9 else None

    lvl_d, lvl_h = level(p_d), level(p_h)
    tone = max(lvl_d, lvl_h, key=lambda x: ORDER[x])
    icon, head, sub = HEADLINE[tone]
    assumed_txt = ", ".join(assumed)

    st.markdown('<div class="section-title">Your screening results</div>'
                '<div class="section-sub">Based on the health information you entered.</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="overall" style="--c:{COLORS[tone]}"><div class="big">{icon}</div><div><h3>{head}</h3>'
                f'<p>{sub}</p><small>Diabetes: {lvl_d} · Heart disease: {lvl_h}</small></div></div>', unsafe_allow_html=True)
    if (not pd.isna(glucose) and glucose >= 200) or (not pd.isna(fasting) and fasting >= 126):
        st.markdown('<div class="alert">A blood sugar value you entered is high. Please see a doctor soon for proper testing.</div>',
                    unsafe_allow_html=True)
    if assumed:
        st.markdown(f'<div class="assume"><b>Typical values were used for:</b> {assumed_txt}. These are not your own '
                    'measurements, so treat the results as a rough guide.</div>', unsafe_allow_html=True)
    if sex == "Male":
        st.markdown('<div class="caution"><b>Note:</b> the diabetes model was trained on records of women, so its score may be '
                    'less reliable for men.</div>', unsafe_allow_html=True)
    st.write("")
    st.markdown(f'<div class="rgrid">{result_card("Diabetes", p_d, lo_d, hi_d, "d", "🩸", "diabetes")}'
                f'{result_card("Heart disease", p_h, lo_h, hi_h, "h", "❤️", "heart")}</div>', unsafe_allow_html=True)
    st.markdown('<div class="note">The likely range comes from 20 models trained on resampled versions of the data. It shows how '
                'much the score depends on the data. It is not a medical confidence interval.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">What influenced your result?</div>'
                '<div class="section-sub">We swapped each of your values for the typical value in the data and measured how '
                'the model score changed. These explanations come from the trained Random Forest models.</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="fgrid">{drivers_card("Diabetes score", "d", drv_d)}'
                f'{drivers_card("Heart disease score", "h", drv_h)}</div>', unsafe_allow_html=True)

    delta_pts = round(p_hi * 100) - round(p_lo * 100)
    if delta_pts > 0:
        reading = f"For this profile, the heart score is about {delta_pts} points higher with high blood sugar."
    elif delta_pts < 0:
        reading = "For this profile, the heart score is not higher with high blood sugar."
    else:
        reading = "For this profile, blood sugar makes little difference to the heart score."
    st.markdown('<div class="section-title">Model sensitivity: blood sugar and heart score</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel"><p>We compare two inputs while keeping everything else you entered the same, to show how the heart '
        f'model responds to fasting blood sugar. {reading} This compares model outputs. It is not proof that one condition '
        'causes the other.</p><table class="sens"><tr><th>Scenario</th><th>Fasting blood sugar</th><th>Heart model score</th><th>Change</th></tr>'
        f'<tr class="{"you" if fbs_flag == 0 else ""}"><td>Baseline</td><td>120 mg/dL or below</td><td>{round(p_lo * 100)} / 100</td><td>-</td></tr>'
        f'<tr class="{"you" if fbs_flag == 1 else ""}"><td>Modified</td><td>Above 120 mg/dL</td><td>{round(p_hi * 100)} / 100</td><td>{delta_pts:+d} points</td></tr></table>'
        f'{bar("Normal blood sugar", p_lo)}{bar("High blood sugar", p_hi)}</div>', unsafe_allow_html=True)
    if whatif is not None:
        base_pts, alt_pts = round(p_d * 100), round(whatif * 100)
        st.markdown('<div class="section-title">Model sensitivity: body weight and diabetes score</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel"><p>We change only your BMI to the upper end of the healthy range (24.9). This compares model '
            'outputs. It is not a promise of what weight loss would do for you.</p><table class="sens"><tr><th>Scenario</th>'
            '<th>BMI</th><th>Diabetes model score</th><th>Change</th></tr>'
            f'<tr><td>Your input</td><td>{bmi}</td><td>{base_pts} / 100</td><td>-</td></tr>'
            f'<tr><td>BMI at 24.9</td><td>24.9</td><td>{alt_pts} / 100</td><td>{alt_pts - base_pts:+d} points</td></tr></table></div>',
            unsafe_allow_html=True)

    focus = []
    if ORDER[lvl_d] >= 1 or (not pd.isna(glucose) and glucose >= 140) or (not pd.isna(fasting) and fasting > 120):
        focus.append(("🩸", "For blood sugar", "Cut sugary drinks and sweets, and choose whole grains, millets and dal over white rice and maida."))
    if ORDER[lvl_h] >= 1 or (not pd.isna(sbp) and sbp >= 130) or (not pd.isna(chol) and chol >= 200):
        focus.append(("❤️", "For your heart", "Eat less salt and fried food, and add vegetables, fruit, nuts and oily fish."))
    if not focus:
        focus.append(("🥗", "Keep it balanced", "Keep eating a mix of vegetables, whole grains, protein and fruit, and limit sweets and fried food."))
    if not pd.isna(dpf) and dpf > float(DM["medians"]["DiabetesPedigreeFunction"]):
        focus.append(("👪", "Family history", "Diabetes in your family can raise your risk, so regular blood sugar checks are a good habit."))
    st.markdown('<div class="section-title">Health recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel">' + "".join(
        f'<div class="focus"><span class="e">{e}</span><div><b>{t}</b>{d}</div></div>' for e, t, d in focus)
        + '<div class="habits">'
          '<div class="habit"><b>🚶 Stay active</b>Aim for at least 150 minutes of moderate activity a week.</div>'
          '<div class="habit"><b>🚭 Avoid tobacco</b>Smoking raises heart risk. Ask for help if you want to quit.</div>'
          '<div class="habit"><b>😴 Sleep well</b>Most adults need about 7 to 9 hours of sleep.</div>'
          '<div class="habit"><b>🩺 Regular check-ups</b>Check blood pressure, sugar and cholesterol regularly.</div></div>'
          '<div class="note" style="margin-top:.8rem">Open <b>← Edit my answers</b> and then the <b>④ Nutrition guide</b> tab for the full food list.</div></div>',
        unsafe_allow_html=True)
    st.markdown('<div class="urgent"><b>Call emergency services (112 in India) right away for:</b>chest pain or pressure, pain '
                'spreading to the arm, jaw or back, shortness of breath, cold sweat or fainting.</div>'
                '<div class="urgent"><b>Get medical help promptly for signs of dangerously high blood sugar:</b>extreme thirst '
                'with frequent urination and unexplained weight loss, nausea or vomiting, fruity-smelling breath, confusion or '
                'unusual drowsiness, or worsening blurred vision.</div>', unsafe_allow_html=True)

    with st.expander("How do the models work?"):
        st.markdown(f'<div class="fgrid">{importance_card("Diabetes model", "d", DM)}'
                    f'{importance_card("Heart model", "h", HM)}</div>', unsafe_allow_html=True)
        st.caption("A Random Forest combines many decision trees. See the Model performance tab for how well each does on test data.")

    summary = (
        "DISEASE RISK PREDICTION - SCREENING SUMMARY\n\n"
        f"Diabetes model score: {p_d * 100:.0f}/100 ({lvl_d}), likely range {lo_d * 100:.0f} to {hi_d * 100:.0f}\n"
        f"Heart disease model score: {p_h * 100:.0f}/100 ({lvl_h}), likely range {lo_h * 100:.0f} to {hi_h * 100:.0f}\n"
        f"Overall: {head}\n\n"
        "Recommendations:\n" + "\n".join(f"- {t}: {d}" for _, t, d in focus) + "\n\n"
        f"Inputs: age {age}, {sex}, BMI {bmi}.\n"
        + (f"Typical values were used for: {assumed_txt}.\n" if assumed else "")
        + "\nThis is an educational screening tool, not a medical diagnosis. Model scores are not exact probabilities. "
          "Please consult a doctor.")
    st.write("")
    b1, b2, b3 = st.columns(3)
    with b1:
        st.download_button("Download my summary", summary, file_name="disease_risk_summary.txt")
    with b2:
        st.button("← Edit my answers", on_click=edit_answers, key="edit_bottom")
    with b3:
        st.button("Start over", on_click=reset_form, key="start_over")
    st.markdown(quote_html(*RESULT_QUOTES[tone]), unsafe_allow_html=True)
    st.markdown('<div class="note">This is an educational screening score from Random Forest models, not a diagnosis and not '
                'the exact chance of getting a disease. The diabetes and heart models are trained on separate datasets, so the '
                'blood sugar comparison shows an association the models learned, not proof of cause. Please consult a doctor '
                'for medical advice.</div>', unsafe_allow_html=True)

else:
    # ============================================================== FORM VIEW
    st.markdown(
        BRAND +
        '<div class="hero"><span class="tag"><i></i>AI-powered health screening</span>'
        '<h1>Understand your health. Take action early.</h1>'
        '<p>Answer a few health questions to explore your estimated diabetes and heart disease risk.</p>'
        '<svg class="ecg" viewBox="0 0 420 44" preserveAspectRatio="none"><path d="M0 24 H70 L82 24 L92 8 L104 40 '
        'L114 24 H190 L202 24 L212 6 L224 42 L234 24 H310 L322 24 L332 10 L344 38 L354 24 H420"/></svg></div>'
        '<div class="chips"><span class="chip c1">🩸 Diabetes screening</span><span class="chip c2">❤️ Heart health screening</span>'
        '<span class="chip c3">Educational tool · Not a medical diagnosis</span></div>', unsafe_allow_html=True)
    if "quote" not in st.session_state:
        st.session_state["quote"] = random.choice(QUOTES)
    st.markdown(quote_html(*st.session_state["quote"]), unsafe_allow_html=True)
    st.markdown('<div class="sec">⚡ Try an example</div>', unsafe_allow_html=True)
    e1, e2, e3 = st.columns([1, 1, 2])
    e1.button("Lower-risk example", on_click=load_example, args=("low",), key="ex_low")
    e2.button("Higher-risk example", on_click=load_example, args=("high",), key="ex_high")
    e3.markdown('<div class="help">Fills the form with sample values so you can see how the app works. '
                'The scores come from the trained models.</div>', unsafe_allow_html=True)
    st.write("")

    tab_basic, tab_lab, tab_fam, tab_food, tab_model = st.tabs(
        ["①  Health details", "②  Lab values", "③  Family history", "④  Nutrition guide", "⑤  Model performance"])

    with tab_basic:
        st.markdown('<div class="help">Things you can find out at home or from a quick check-up. Tick "I don\'t know" '
                    'when you are unsure: a typical value is used and your result is marked.</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec">👤 Personal information</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        age = c1.number_input("Age (years, 21 or older)", 21, 90, key="age")
        sex = c2.selectbox("Sex", ["Female", "Male"], key="sex")
        height = c3.number_input("Height (cm)", 120, 220, key="height")
        weight = c4.number_input("Weight (kg)", 30, 200, key="weight")
        bmi = round(weight / ((height / 100) ** 2), 1)
        st.markdown(f'<div class="bmi">Your BMI: {bmi} · {bmi_category(bmi)}</div>', unsafe_allow_html=True)
        if sex == "Female":
            st.number_input("Number of pregnancies", 0, 20, key="preg", help="Enter 0 if not applicable.")
        else:
            st.markdown('<div class="notice"><b>Note for men</b>The diabetes model was trained on records of women, so its '
                        'score may be less reliable. The heart model uses both sexes.</div>', unsafe_allow_html=True)

        st.markdown('<div class="sec">🩺 Vital signs</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with_unknown(c1, "Blood pressure: top number (mmHg)", 80, 220, "sbp", help="Systolic pressure. Typical healthy reading is around 120.")
        with_unknown(c2, "Blood pressure: bottom number (mmHg)", 30, 140, "dbp", help="Diastolic pressure. Typical healthy reading is around 80.")
        with_unknown(c3, "Cholesterol (mg/dL)", 100, 600, "chol", help="Total cholesterol from a blood test.")

        st.markdown('<div class="sec">🍬 Blood sugar</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with_unknown(c1, "Fasting blood sugar (mg/dL)", 40, 400, "fbsv",
                     help="Blood sugar after not eating for 8 hours. Above 120 counts as high for the heart model.")
        ogtt = c2.radio("Have you had a glucose tolerance test (OGTT)?", ["No or not sure", "Yes"], key="ogtt", horizontal=True,
                        help="A test where blood sugar is measured 2 hours after drinking a glucose solution.")
        if ogtt == "Yes":
            c2.number_input("Blood glucose 2 hours after the glucose drink (mg/dL)", 40, 300, key="glucose",
                            help="Use the 2-hour value from the test report. The diabetes model needs this specific test.")
        else:
            c2.caption("The diabetes model uses the 2-hour glucose tolerance test value. A fasting or random reading is a "
                       "different test, so a typical value is used and your result is marked.")

        st.markdown('<div class="sec">🫀 Chest symptoms</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        cp_label = c1.selectbox("Chest pain level", list(CHEST_PAIN), key="cp")
        c1.caption(f"Medical type: {CHEST_PAIN_TYPE[cp_label]}")
        c2.selectbox("Chest pain during exercise?", ["No", "Yes"], key="exang")

    with tab_lab:
        st.markdown('<div class="notice"><b>Optional clinical measurements</b>These come from lab tests, scans or a clinic '
                    'visit. Most people do not know them, so "I don\'t know" is the default. Unknown values are replaced with '
                    'the typical value from the training data, and your results are clearly marked.</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec">🩸 For the diabetes model</div>', unsafe_allow_html=True)
        a, b = st.columns(2)
        with_unknown(a, "Insulin, 2 hours after glucose drink (mu U/ml)", 0, 900, "insulin", help="Serum insulin from a lab test.")
        with_unknown(b, "Triceps skin fold thickness (mm)", 5, 100, "skin", help="Measured with a skin-fold caliper at the back of the upper arm.")
        st.markdown('<div class="sec">❤️ For the heart model</div>', unsafe_allow_html=True)
        a, b = st.columns(2)
        with_unknown(a, "Highest heart rate in an exercise test (bpm)", 60, 220, "maxhr")
        with_unknown(b, "ST depression (oldpeak)", 0.0, 7.0, "oldpeak", help="A measurement from an exercise ECG test.")
        a, b = st.columns(2)
        a.selectbox("Resting ECG result", [UNKNOWN] + list(ECG), key="ecg")
        b.selectbox("Exercise ECG slope", [UNKNOWN] + list(SLOPE), key="slope")
        a, b = st.columns(2)
        a.selectbox("Blocked vessels seen on scan (0 to 3)", [UNKNOWN, "0", "1", "2", "3"], key="ca")
        b.selectbox("Thalassemia blood test", [UNKNOWN] + list(THAL), key="thal")

    with tab_fam:
        st.markdown('<div class="notice"><b>Family history of diabetes</b>Diabetes in blood relatives is an input of the '
                    'diabetes model. Add the relatives you know about, <b>including relatives who do not have diabetes</b>. '
                    'If you skip this tab, a typical value is used and your result is marked.</div>', unsafe_allow_html=True)
        with st.form("relative_form", clear_on_submit=True):
            c1, c2, c3 = st.columns([2, 1.3, 1.3])
            rel = c1.selectbox("Relative", list(RELATIVES))
            has_dm = c2.selectbox("Has or had diabetes?", ["No", "Yes"])
            rage = c3.number_input("Age", 1, 100, 50,
                                   help="If they have diabetes: the age when it was diagnosed. If not: their current age.")
            added = st.form_submit_button("Add relative")
        if added:
            st.session_state.setdefault("relatives", []).append({"rel": rel, "dm": has_dm == "Yes", "age": int(rage)})
        rels = st.session_state.get("relatives", [])
        if rels:
            rows = "".join(
                f'<tr><td>{r["rel"]}</td><td>{"Diabetes, diagnosed at " + str(r["age"]) if r["dm"] else "No diabetes, age " + str(r["age"])}</td>'
                f'<td>{RELATIVES[r["rel"]] * 100:g}%</td></tr>' for r in rels)
            st.markdown('<table class="sens"><tr><th>Relative</th><th>Details</th><th>Genes shared</th></tr>' + rows + '</table>',
                        unsafe_allow_html=True)
            r1, r2 = st.columns(2)
            r1.button("Remove last", on_click=remove_last_relative)
            r2.button("Clear list", on_click=clear_relatives)
            fam = compute_dpf(rels)
            typ = float(DM["medians"]["DiabetesPedigreeFunction"])
            word = "above" if fam > typ * 1.1 else "below" if fam < typ * 0.9 else "close to"
            st.markdown(f'<div class="bmi">Your family history score: {fam:.2f} · {word} the typical value ({typ:.2f})</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="help">No relatives added yet. A typical family history score will be used.</div>',
                        unsafe_allow_html=True)
        with st.expander("How is the family history score calculated?"):
            st.markdown(
                "It follows the published *diabetes pedigree function* used to build the diabetes dataset (Smith et al., 1988):\n\n"
                "`score = (20 + sum of share × (88 − age at diagnosis) for relatives with diabetes) ÷ "
                "(50 + sum of share × (age − 14) for relatives without diabetes)`\n\n"
                "*Share* is the fraction of genes you have in common with the relative: 0.5 for parents and full siblings, "
                "0.25 for half siblings, grandparents, aunts and uncles, and 0.125 for half aunts or uncles and first cousins. "
                "Terms below zero count as zero.")

    with tab_food:
        st.markdown('<div class="help">What to eat and what to limit for a healthy sugar level and a healthy heart. '
                    'This is general guidance, not a personal diet plan.</div>', unsafe_allow_html=True)
        st.markdown(plate_html(), unsafe_allow_html=True)
        fc1, fc2 = st.columns(2)
        for col, key in ((fc1, "diabetes"), (fc2, "heart")):
            f = FOOD[key]
            g1, g2 = ("#2563EB", "#06B6D4") if key == "diabetes" else ("#E11D48", "#F97316")
            with col:
                st.markdown(f'<div class="fhead" style="--g1:{g1};--g2:{g2}"><span class="ico" style="--g1:{g1};--g2:{g2}">{f["icon"]}</span>'
                            f'<div class="rtitle">{f["title"]}</div></div>', unsafe_allow_html=True)
                for cls, name, items in f["groups"]:
                    with st.expander(f"{GROUP_ICON[cls]} {name}", expanded=(cls == "good")):
                        st.markdown("\n".join(f"- {i}" for i in items))
        st.caption("Eat regular meals, watch portion sizes, drink plenty of water and stay active. If you are pregnant, "
                   "have kidney disease or take medicines for diabetes or blood pressure, follow your doctor's or dietitian's advice.")

    with tab_model:
        st.markdown('<div class="help">How well each Random Forest performs on records it has never seen (a held-out 20% test set). '
                    'Zero readings in the diabetes data are treated as missing and filled with the training median.</div>',
                    unsafe_allow_html=True)
        metrics_block("🩸 Diabetes model", DM)
        metrics_block("❤️ Heart model", HM)
        st.markdown('<div class="note">Limits: small datasets, specific patient groups, and scores are not exact probabilities.</div>',
                    unsafe_allow_html=True)

    st.write("")
    st.button("Check my risk", type="primary", on_click=show_results, key="check")

st.markdown('<div class="foot"><b>Disease Risk Prediction</b> · Diabetes and heart screening · Random Forest · Streamlit · v2.0<br>'
            'Artificial Intelligence and Data Science · HNN College of Engineering<br>'
            'Educational screening tool, not a medical diagnosis. This app has no accounts or database and does not '
            'store your answers itself.</div>', unsafe_allow_html=True)
