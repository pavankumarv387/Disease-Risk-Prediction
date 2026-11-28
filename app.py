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

st.set_page_config(page_title="HealthGuard AI | Diabetes Risk Check", page_icon="🩺", layout="wide")

COLUMNS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
           "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
ZERO_MEANS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
FRIENDLY = {
    "Pregnancies": "Pregnancies", "Glucose": "Blood glucose", "BloodPressure": "Blood pressure",
    "SkinThickness": "Skin fold thickness", "Insulin": "Insulin level", "BMI": "BMI",
    "DiabetesPedigreeFunction": "Family history score", "Age": "Age",
}

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
    "Low": "Keep a balanced diet and stay active.",
    "Moderate": "Cut down sugary drinks, walk daily and recheck your blood sugar in a few months.",
    "High": "Please get a fasting blood sugar or HbA1c test and talk to a doctor.",
}
HEADLINE = {
    "Low": ("🟢", "Good news: low diabetes risk score", "Keep up your healthy habits and regular check-ups."),
    "Moderate": ("🟠", "Some diabetes risk signs were found", "A check-up is a good idea. See the recommendations below."),
    "High": ("🔴", "Higher diabetes risk score: please see a doctor soon",
             "This tool cannot diagnose. A doctor can run the right tests."),
}
# Share of genes held in common with the relative (as used in the published diabetes pedigree function).
RELATIVES = {
    "Mother": 0.5, "Father": 0.5, "Full brother or sister": 0.5,
    "Half brother or sister": 0.25, "Grandparent": 0.25, "Aunt or uncle": 0.25,
    "Half aunt or half uncle": 0.125, "First cousin": 0.125,
}
FOOD = [
    ("good", "🟢", "Eat more", [
        "Non-starchy vegetables: spinach, beans, gourds, cabbage, cauliflower",
        "Dal, sprouts, chickpeas and other legumes",
        "Whole grains and millets: ragi, jowar, oats, brown rice",
        "Lean protein: fish, eggs, skinless chicken, paneer",
        "Unsweetened curd or buttermilk, and a small handful of nuts",
    ]),
    ("warn", "🟠", "In moderation", [
        "Rice, chapati and bread: keep portions small",
        "Whole fruits (not juice): one serving at a time",
        "Potato, sweet potato and other starchy vegetables",
        "Honey and jaggery: they are still sugar",
    ]),
    ("bad", "🔴", "Limit or avoid", [
        "Sugary drinks, packaged juices, sweets and desserts",
        "Maida, white bread and bakery items",
        "Deep-fried snacks and packaged foods",
        "Extra sugar in tea and coffee, and alcohol",
    ]),
]

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

.brand {display: flex; align-items: center; gap: .8rem; margin-bottom: 1rem;}
.logo {display: inline-grid; place-items: center; width: 2.6rem; height: 2.6rem; border-radius: 14px; font-size: 1.3rem;
  background: linear-gradient(135deg, #1D4ED8, #0D9488);}
.brand b {display: block; font-size: 1.2rem; font-weight: 800; letter-spacing: -0.01em; line-height: 1.1;}
.brand small {opacity: .7; font-weight: 600;}
.brand .bt {margin-left: auto; padding: .3rem .8rem; border-radius: 99px; font-size: .78rem; font-weight: 700;
  border: 1px solid rgba(13,148,136,.5); background: rgba(13,148,136,.14);}

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
.c2 {background: linear-gradient(90deg, #0D9488, #22C55E);}
.c3 {background: rgba(128,128,128,.35);}

.quote {margin: 1rem 0 .5rem; padding: .9rem 1.2rem .95rem 1.4rem; border-radius: 14px; border-left: 5px solid #2563EB;
  background: linear-gradient(120deg, rgba(37,99,235,.12), rgba(13,148,136,.12));
  font-style: italic; line-height: 1.5; font-size: 1.02rem;}
.quote small {display: block; font-style: normal; opacity: .7; margin-top: .3rem; font-size: .85rem; font-weight: 600;}

.help {margin: .2rem 0 .8rem; font-size: .9rem; opacity: .78; line-height: 1.5;}
.sec {display: flex; align-items: center; gap: .5rem; margin: 1.1rem 0 .5rem; font-weight: 800; font-size: 1.05rem;}
.sec::after {content: ""; flex: 1; height: 1px; background: rgba(128,128,128,.3);}
.notice {margin: .2rem 0 1rem; padding: .85rem 1rem; border-radius: 14px; font-size: .9rem; line-height: 1.5;
  border: 1px solid rgba(37,99,235,.35); background: rgba(37,99,235,.10);}
.notice b {display: block; margin-bottom: .15rem;}
.stTabs [data-baseweb="tab"] {font-weight: 700; font-size: 1rem;}
.stTabs [data-baseweb="tab-panel"] {margin-top: .8rem; padding: 1rem 1.2rem .6rem; border-radius: 18px;
  border: 1px solid rgba(37,99,235,.25); background: rgba(37,99,235,.05);}
.bmi {display: inline-block; margin: .1rem 0 .4rem; padding: .35rem .8rem; border-radius: 99px; font-weight: 700; font-size: .88rem;
  background: rgba(37,99,235,.14); border: 1px solid rgba(37,99,235,.35);}

div[data-testid="stButton"], div[data-testid="stDownloadButton"] {width: 100%;}
div[data-testid="stButton"] button, div[data-testid="stDownloadButton"] button {
  width: 100%; border-radius: 12px; font-weight: 800; padding: .8rem 1rem; font-size: 1.06rem; transition: transform .15s ease, box-shadow .15s ease;}
div[data-testid="stButton"] button[kind="primary"], div[data-testid="stButton"] button[data-testid="stBaseButton-primary"] {
  background-image: linear-gradient(100deg, #1D4ED8, #0D9488); background-color: #1D4ED8; color: #fff; border: none;
  box-shadow: 0 14px 28px -14px rgba(29,78,216,.9);}
div[data-testid="stButton"] button[kind="primary"]:hover, div[data-testid="stButton"] button[data-testid="stBaseButton-primary"]:hover {
  transform: translateY(-2px); color: #fff; box-shadow: 0 18px 32px -14px rgba(13,148,136,.9);}

.section-title {font-size: 1.3rem; font-weight: 800; margin: 2.4rem 0 .3rem; letter-spacing: -0.01em; display: flex; align-items: center; gap: .6rem;}
.section-title::before {content: ""; width: 5px; height: 1.3rem; border-radius: 4px; background: linear-gradient(180deg, #1D4ED8, #0D9488);}
.section-sub {margin: 0 0 1rem; opacity: .72; font-size: .92rem; line-height: 1.5;}
.overall {display: flex; align-items: center; gap: 1rem; padding: 1.3rem 1.4rem; border-radius: 20px; border-left: 8px solid var(--c);
  background: rgba(128,128,128,.12); background: color-mix(in srgb, var(--c) 15%, transparent);}
.overall .big {font-size: 2.3rem; line-height: 1;}
.overall h3 {margin: 0; padding: 0; font-size: 1.35rem; font-weight: 800; line-height: 1.25;}
.overall p {margin: .3rem 0 0; opacity: .85; line-height: 1.5;}
.assume {margin-top: .8rem; padding: .7rem 1rem; border-radius: 12px; font-size: .88rem; line-height: 1.5;
  border: 1px dashed rgba(217,119,6,.6); background: rgba(217,119,6,.10);}
.alert {margin-top: .8rem; padding: .7rem 1rem; border-radius: 12px; font-size: .9rem; line-height: 1.5; font-weight: 700;
  border: 1px solid rgba(220,38,38,.5); background: rgba(220,38,38,.12);}
.rgrid {display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem;}
.rcard {position: relative; overflow: hidden; padding: 1.5rem 1.3rem 1.3rem; border-radius: 22px; border: 1px solid rgba(128,128,128,.25);
  --g1: #2563EB; --g2: #06B6D4;
  background: linear-gradient(160deg, color-mix(in srgb, var(--g1) 11%, transparent), color-mix(in srgb, var(--g2) 6%, transparent));
  box-shadow: 0 16px 34px -24px var(--g1);}
.rcard::before {content: ""; position: absolute; left: 0; right: 0; top: 0; height: 6px; background: linear-gradient(90deg, var(--g1), var(--g2));}
.rhead {display: flex; align-items: center; gap: .7rem; margin-bottom: 1rem;}
.ico {display: inline-grid; place-items: center; width: 2.3rem; height: 2.3rem; border-radius: 12px; font-size: 1.1rem;
  background: linear-gradient(135deg, #2563EB, #06B6D4);}
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
.tip {margin-top: .9rem; padding: .7rem .9rem; border-radius: 12px; font-size: .88rem; line-height: 1.45; background: rgba(128,128,128,.12);}
.tip b {display: block; font-size: .78rem; opacity: .7; margin-bottom: .15rem;}

.panel {border: 1px solid rgba(37,99,235,.28); border-radius: 20px; padding: 1.25rem 1.35rem;
  background: linear-gradient(135deg, rgba(37,99,235,.09), rgba(13,148,136,.09));}
.panel p {margin: 0 0 1rem; line-height: 1.55;}
.barrow {display: grid; grid-template-columns: 10rem 1fr 3.2rem; align-items: center; gap: .7rem; margin: .55rem 0;}
.barl {font-size: .9rem; opacity: .9; font-weight: 600;}
.barv {font-weight: 800; text-align: right; font-size: .92rem;}
.track {height: 12px; border-radius: 99px; background: rgba(128,128,128,.25); overflow: hidden;}
.fill {height: 100%; border-radius: 99px;}
.fcard {border: 1px solid rgba(128,128,128,.25); border-radius: 22px; padding: 1.3rem 1.3rem; background: rgba(128,128,128,.06);}
.fcard h4 {margin: 0 0 .2rem; font-size: 1.05rem; font-weight: 800;}
.fcard small {display: block; opacity: .65; margin-bottom: .6rem;}
.fcard.static .fill {background: linear-gradient(90deg, #2563EB, #06B6D4);}
.sens {width: 100%; border-collapse: collapse; margin-top: .4rem; font-size: .92rem;}
.sens th, .sens td {padding: .55rem .7rem; text-align: left; border-bottom: 1px solid rgba(128,128,128,.25);}
.sens th {font-size: .78rem; opacity: .7;}
@media (max-width: 560px) {.barrow {grid-template-columns: 1fr 3.2rem;} .barl {grid-column: 1 / -1;}}

.habits {display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: .8rem; margin-top: .9rem;}
.habit {padding: .8rem 1rem; border-radius: 14px; background: rgba(128,128,128,.10); font-size: .9rem; line-height: 1.45;}
.habit b {display: block; margin-bottom: .1rem;}
.focus {display: flex; gap: .8rem; align-items: flex-start; margin: .55rem 0; line-height: 1.5;}
.focus .e {font-size: 1.4rem; line-height: 1.2;}
.focus b {display: block;}

.plate {display: flex; align-items: center; gap: 1.4rem; flex-wrap: wrap; padding: 1.1rem 1.3rem; margin-bottom: 1rem; border-radius: 18px;
  border: 1px solid rgba(128,128,128,.25); background: rgba(128,128,128,.07);}
.disc {flex: none; width: 118px; height: 118px; border-radius: 50%; box-shadow: 0 0 0 6px rgba(128,128,128,.3);
  background: conic-gradient(#16A34A 0 50%, #F59E0B 0 75%, #2563EB 0 100%);}
.plate ul {list-style: none; margin: 0; padding: 0; flex: 1 1 220px;}
.plate li {display: flex; align-items: center; gap: .6rem; margin: .35rem 0; font-size: .93rem; line-height: 1.35; font-weight: 600;}
.plate li i {flex: none; width: 14px; height: 14px; border-radius: 4px;}
.plate h4 {margin: 0 0 .4rem; font-size: 1rem; font-weight: 800;}

.mgrid {display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: .8rem; margin: .4rem 0 1rem;}
.mtile {padding: 1rem 1rem; border-radius: 16px; border: 1px solid rgba(128,128,128,.25); background: rgba(128,128,128,.07); text-align: center;}
.mtile b {display: block; font-size: 1.7rem; font-weight: 800; line-height: 1.1;
  background: linear-gradient(100deg, #2563EB, #0D9488); -webkit-background-clip: text; background-clip: text; color: transparent;}
.mtile span {font-size: .82rem; opacity: .75; font-weight: 600;}
.cm {border-collapse: collapse; margin: .4rem 0 1rem; font-size: .92rem;}
.cm th, .cm td {padding: .6rem 1rem; border: 1px solid rgba(128,128,128,.3); text-align: center;}
.cm th {font-size: .8rem; opacity: .75;}
.cm td b {font-size: 1.2rem;}

.note {margin-top: 1.3rem; font-size: .84rem; opacity: .68; line-height: 1.5;}
.foot {margin-top: 2.4rem; padding-top: 1.1rem; border-top: 1px solid rgba(128,128,128,.25); text-align: center; font-size: .8rem; opacity: .65; line-height: 1.6;}

.rangetxt {text-align: center; font-size: .85rem; font-weight: 700; opacity: .85; margin-top: .2rem;}
.rangebar {position: relative; height: 10px; border-radius: 99px; background: rgba(128,128,128,.25); margin: .5rem auto 1.1rem; max-width: 260px;}
.rangebar .seg {position: absolute; top: 0; bottom: 0; border-radius: 99px; background: var(--c); opacity: .6;}
.rangebar .mark {position: absolute; top: -4px; width: 4px; height: 18px; border-radius: 2px; background: #fff; transform: translateX(-50%);
  box-shadow: 0 0 0 1px rgba(0,0,0,.55);}
.urgent {margin-top: 1rem; padding: 1rem 1.2rem; border-radius: 16px; border: 1px solid rgba(220,38,38,.5);
  background: rgba(220,38,38,.10); font-size: .92rem; line-height: 1.55;}
.urgent b {display: block; margin-bottom: .2rem;}
.examples {display: flex; align-items: center; gap: .6rem; flex-wrap: wrap;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ model (trained when the app starts)
@st.cache_resource(show_spinner="Training the Random Forest model...")
def load_model():
    path = Path(__file__).parent / "diabetes.csv"
    if not path.exists():
        path = Path("diabetes.csv")
    if not path.exists():
        return "missing"
    df = pd.read_csv(path)
    if df.shape[1] != 9:
        return "columns"
    df.columns = COLUMNS  # assigned by position
    df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=["Outcome"])
    for c in ZERO_MEANS_MISSING:
        df[c] = df[c].replace(0, np.nan)  # 0 is not a real reading, treat as missing
    X, y = df.drop(columns="Outcome"), df["Outcome"].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    def new_model(trees=200, seed=42):
        return make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=trees, random_state=seed))

    model = new_model().fit(Xtr, ytr)
    pred, prob = model.predict(Xte), model.predict_proba(Xte)[:, 1]
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    cv = cross_val_score(new_model(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="accuracy").mean()
    # 20 smaller forests trained on resampled copies of the training data, used to show a likely score range.
    rng = np.random.default_rng(42)
    boots = []
    for i in range(20):
        idx = rng.integers(0, len(Xtr), len(Xtr))
        boots.append(new_model(60, i).fit(Xtr.iloc[idx], ytr.iloc[idx]))
    return {
        "model": model, "boots": boots,
        "medians": pd.Series(model[0].statistics_, index=X.columns),
        "importance": pd.Series(model[-1].feature_importances_, index=X.columns).sort_values(ascending=False),
        "metrics": {"Accuracy": accuracy_score(yte, pred), "Precision": precision_score(yte, pred),
                    "Recall": recall_score(yte, pred), "F1 score": f1_score(yte, pred),
                    "ROC-AUC": roc_auc_score(yte, prob), "5-fold CV accuracy": cv},
        "cm": (tn, fp, fn, tp), "n": len(df), "n_test": len(yte), "pos_rate": float(y.mean()),
    }


M = load_model()
if isinstance(M, str):
    msg = ("Add your <b>diabetes.csv</b> file to the same folder as app.py in your GitHub repository, then reload this page."
           if M == "missing" else "diabetes.csv should have 9 columns: Pregnancies, Glucose, BloodPressure, SkinThickness, "
                                  "Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome.")
    st.markdown(f'<div class="notice"><b>Dataset problem</b>{msg}</div>', unsafe_allow_html=True)
    st.stop()
MODEL, MED = M["model"], M["medians"]


def level(p):
    return "Low" if p < 0.33 else "Moderate" if p < 0.66 else "High"


def quote_html(text, author):
    return f'<div class="quote">&ldquo;{text}&rdquo;<small>{author}</small></div>'


def result_card(p, lo, hi):
    score, lvl = round(p * 100), level(p)
    return (f'<div class="rcard"><div class="rhead"><span class="ico">🩸</span>'
            f'<div><div class="rtitle">Diabetes</div><div class="rsub">Model score</div></div>'
            f'<span class="pill" style="--c:{COLORS[lvl]}">{lvl}</span></div>'
            f'<div class="ring" style="--p:{score};--c:{COLORS[lvl]}"><div class="ringtxt"><b>{score}</b><span>out of 100</span></div></div>'
            f'<div class="lvl" style="--c:{COLORS[lvl]}">{lvl} risk category</div>'
            f'<div class="rangetxt">Likely range: {round(lo * 100)} to {round(hi * 100)}</div>'
            f'<div class="rangebar" style="--c:{COLORS[lvl]}"><div class="seg" style="left:{lo * 100:.0f}%;width:{max((hi - lo) * 100, 2):.0f}%"></div>'
            f'<div class="mark" style="left:{score}%"></div></div>'
            f'<div class="rtext">{MEANING[lvl]}</div>'
            f'<div class="tip"><b>What you can do</b>{TIPS[lvl]}</div></div>')


def predict_range(vals, p):
    df = pd.DataFrame([vals], columns=COLUMNS[:-1])
    scores = [float(m.predict_proba(df)[0, 1]) for m in M["boots"]]
    lo, hi = np.percentile(scores, [10, 90])
    return min(float(lo), p), max(float(hi), p)


def predict(vals):
    return float(MODEL.predict_proba(pd.DataFrame([vals], columns=COLUMNS[:-1]))[0, 1])


def drivers(vals, base, top=5):
    """Swap each of your values for the typical (median) value and see how the model score changes."""
    out = []
    for k in COLUMNS[:-1]:
        if pd.isna(vals[k]):
            continue
        alt = dict(vals)
        alt[k] = float(MED[k])
        out.append((k, base - predict(alt)))
    out = [x for x in sorted(out, key=lambda x: -abs(x[1])) if abs(x[1]) >= 0.005]
    return out[:top]


def drivers_card(items):
    if not items:
        body = '<div class="rsub">No single value stood out compared with a typical person in the data.</div>'
    else:
        mx = max(abs(d) for _, d in items) or 1
        body = "".join(
            f'<div class="barrow"><span class="barl">{"▲" if d > 0 else "▼"} {FRIENDLY[k]}</span>'
            f'<div class="track"><div class="fill" style="width:{abs(d) / mx * 100:.0f}%;background:{"#DC2626" if d > 0 else "#16A34A"}"></div></div>'
            f'<span class="barv">{d * 100:+.0f}</span></div>' for k, d in items)
    return ('<div class="fcard"><h4>What influenced your score?</h4>'
            '<small>▲ raises your score · ▼ lowers it (points out of 100), compared with a typical person in the data</small>'
            f'{body}</div>')


def importance_card(top=8):
    imp = M["importance"].head(top)
    mx = imp.max() or 1
    rows = "".join(
        f'<div class="barrow"><span class="barl">{FRIENDLY[n]}</span>'
        f'<div class="track"><div class="fill" style="width:{v / mx * 100:.0f}%"></div></div>'
        f'<span class="barv">{v * 100:.0f}%</span></div>' for n, v in imp.items())
    return f'<div class="fcard static"><h4>Inputs the model relies on most</h4><small>Across all training data, not for you alone</small>{rows}</div>'


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


DEFAULTS = {
    "age": 30, "preg": 0, "height": 160, "weight": 60,
    "ogtt": "No or not sure", "glucose": int(round(MED["Glucose"])),
    "dbp": int(round(MED["BloodPressure"])), "dbp_unk": False,
    "insulin": int(round(MED["Insulin"])), "insulin_unk": True,
    "skin": int(round(MED["SkinThickness"])), "skin_unk": True,
}
EXAMPLES = {
    "low": {"inputs": {"age": 28, "preg": 0, "height": 165, "weight": 58, "ogtt": "Yes", "glucose": 92,
                       "dbp": 68, "dbp_unk": False, "insulin_unk": True, "skin_unk": True},
            "relatives": [{"rel": "Mother", "dm": False, "age": 54}, {"rel": "Father", "dm": False, "age": 58}]},
    "high": {"inputs": {"age": 49, "preg": 5, "height": 158, "weight": 82, "ogtt": "Yes", "glucose": 178,
                        "dbp": 88, "dbp_unk": False, "insulin": 210, "insulin_unk": False, "skin": 40, "skin_unk": False},
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


BRAND = (
    '<div class="brand"><span class="logo">🛡️</span><div><b>HealthGuard AI</b><small>Diabetes Risk Prediction</small></div>'
    '<span class="bt">AI-powered screening</span></div>')

if st.session_state.get("view") == "results" and "saved" in st.session_state:
    st.markdown(BRAND, unsafe_allow_html=True)
    if st.session_state.pop("scroll_top", False):
        components.html(
            "<script>const d = window.parent.document;"
            "const el = d.querySelector('[data-testid=\"stMain\"]') || d.querySelector('section.main') || d.documentElement;"
            "try { el.scrollTo({top: 0}); window.parent.scrollTo({top: 0}); } catch (e) {}</script>", height=0)
    st.button("← Edit my answers", on_click=edit_answers, key="edit_top")

    S = st.session_state["saved"]
    rels = st.session_state.get("relatives", [])
    age, preg = S["age"], S["preg"]
    bmi = round(S["weight"] / ((S["height"] / 100) ** 2), 1)
    glucose = S["glucose"] if S["ogtt"] == "Yes" else np.nan
    dbp = np.nan if S["dbp_unk"] else S["dbp"]
    insulin = np.nan if S["insulin_unk"] else S["insulin"]
    skin = np.nan if S["skin_unk"] else S["skin"]
    dpf = compute_dpf(rels) if rels else np.nan
    assumed = [name for cond, name in (
        (S["ogtt"] != "Yes", "blood glucose (no glucose tolerance test)"), (S["dbp_unk"], "blood pressure"),
        (S["insulin_unk"], "insulin level"), (S["skin_unk"], "skin fold thickness"),
        (not rels, "family history score")) if cond]

    vals = {"Pregnancies": float(preg), "Glucose": glucose, "BloodPressure": dbp, "SkinThickness": skin,
            "Insulin": insulin, "BMI": bmi, "DiabetesPedigreeFunction": dpf, "Age": float(age)}
    with st.spinner("Analyzing your inputs..."):
        p = predict(vals)
        lo, hi = predict_range(vals, p)
        drv = drivers(vals, p)
        whatif = predict({**vals, "BMI": 24.9}) if bmi > 24.9 else None
    lvl = level(p)
    icon, head, sub = HEADLINE[lvl]
    assumed_txt = ", ".join(assumed)

    st.markdown('<div class="section-title">Your screening result</div>'
                '<div class="section-sub">Based on the health information you entered.</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="overall" style="--c:{COLORS[lvl]}"><div class="big">{icon}</div><div><h3>{head}</h3>'
                f'<p>{sub}</p></div></div>', unsafe_allow_html=True)
    if not np.isnan(glucose) and glucose >= 200:
        st.markdown('<div class="alert">The glucose value you entered is high. Please see a doctor soon for proper testing.</div>',
                    unsafe_allow_html=True)
    if assumed:
        st.markdown(f'<div class="assume"><b>Typical values were used for:</b> {assumed_txt}. These are not your own '
                    'measurements, so treat the result as a rough guide.</div>', unsafe_allow_html=True)
    st.write("")
    st.markdown(f'<div class="rgrid">{result_card(p, lo, hi)}{drivers_card(drv)}</div>', unsafe_allow_html=True)
    st.markdown('<div class="note">The likely range comes from 20 models trained on resampled versions of the data. It shows how '
                'much the score depends on the data. It is not a medical confidence interval.</div>', unsafe_allow_html=True)

    if whatif is not None:
        base_pts, alt_pts = round(p * 100), round(whatif * 100)
        st.markdown('<div class="section-title">Model sensitivity: body weight</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel"><p>We change only your BMI to the upper end of the healthy range (24.9) and keep everything '
            'else the same, to show how the model responds. This compares model outputs. It is not a promise of what weight '
            'loss would do for you.</p><table class="sens"><tr><th>Scenario</th><th>BMI</th><th>Model score</th><th>Change</th></tr>'
            f'<tr><td>Your input</td><td>{bmi}</td><td>{base_pts} / 100</td><td>-</td></tr>'
            f'<tr><td>BMI at 24.9</td><td>24.9</td><td>{alt_pts} / 100</td><td>{alt_pts - base_pts:+d} points</td></tr></table></div>',
            unsafe_allow_html=True)

    focus = (("🩸", "For blood sugar", "Cut sugary drinks and sweets, and choose whole grains, millets and dal over white rice and maida.")
             if ORDER[lvl] >= 1 or (not np.isnan(glucose) and glucose >= 140)
             else ("🥗", "Keep it balanced", "Keep eating a mix of vegetables, whole grains, protein and fruit, and limit sweets and fried food."))
    family_line = ""
    if not np.isnan(dpf) and dpf > float(MED["DiabetesPedigreeFunction"]):
        family_line = ('<div class="focus"><span class="e">👪</span><div><b>Family history</b>Diabetes in your family can raise '
                       'your risk, so regular blood sugar checks are a good habit.</div></div>')
    st.markdown('<div class="section-title">Health recommendations</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="panel"><div class="focus"><span class="e">{focus[0]}</span><div><b>{focus[1]}</b>{focus[2]}</div></div>{family_line}'
        '<div class="habits">'
        '<div class="habit"><b>🚶 Stay active</b>Aim for at least 150 minutes of moderate activity a week.</div>'
        '<div class="habit"><b>🚭 Avoid tobacco</b>Smoking raises the risk of many diseases.</div>'
        '<div class="habit"><b>😴 Sleep well</b>Most adults need about 7 to 9 hours of sleep.</div>'
        '<div class="habit"><b>🩺 Regular check-ups</b>Check blood sugar and blood pressure regularly.</div></div>'
        '<div class="note" style="margin-top:.8rem">Open <b>← Edit my answers</b> and then the <b>④ Nutrition guide</b> tab for the full food list.</div></div>',
        unsafe_allow_html=True)
    st.markdown('<div class="urgent"><b>Get medical help promptly if you have:</b>extreme thirst with frequent urination and '
                'unexplained weight loss, nausea or vomiting, fruity-smelling breath, confusion or unusual drowsiness, or blurred '
                'vision that is getting worse. These can be signs of dangerously high blood sugar.</div>', unsafe_allow_html=True)

    with st.expander("How does the model work?"):
        st.markdown(f'<div class="rgrid">{importance_card()}</div>', unsafe_allow_html=True)
        st.caption("A Random Forest combines many decision trees. See the Model performance tab for how well it does on test data.")

    summary = (
        "HEALTHGUARD AI - DIABETES RISK SCREENING SUMMARY\n\n"
        f"Diabetes model score: {p * 100:.0f}/100 ({lvl}), likely range {lo * 100:.0f} to {hi * 100:.0f}\nOverall: {head}\n\n"
        f"Inputs: age {age}, pregnancies {preg}, BMI {bmi}"
        + ("" if np.isnan(glucose) else f", glucose {glucose} mg/dL")
        + ("" if np.isnan(dbp) else f", diastolic BP {dbp} mmHg")
        + ("" if np.isnan(dpf) else f", family history score {dpf:.2f}") + ".\n"
        + (f"Typical values were used for: {assumed_txt}.\n" if assumed else "")
        + f"\nRecommendation: {focus[2]}\n\n"
          "This is an educational screening tool, not a medical diagnosis. Model scores are not exact probabilities. "
          "Please consult a doctor.")
    st.write("")
    b1, b2, b3 = st.columns(3)
    with b1:
        st.download_button("Download my summary", summary, file_name="healthguard_diabetes_summary.txt")
    with b2:
        st.button("← Edit my answers", on_click=edit_answers, key="edit_bottom")
    with b3:
        st.button("Start over", on_click=reset_form, key="start_over")
    st.markdown(quote_html(*RESULT_QUOTES[lvl]), unsafe_allow_html=True)
    st.markdown('<div class="note">This is an educational screening score from a Random Forest model, not a diagnosis and not '
                'the exact chance of getting diabetes. Please consult a doctor for medical advice.</div>', unsafe_allow_html=True)
else:
    # ------------------------------------------------------------------ header
    st.markdown(
        '<div class="brand"><span class="logo">🛡️</span><div><b>HealthGuard AI</b><small>Diabetes Risk Prediction</small></div>'
        '<span class="bt">AI-powered screening</span></div>'
        '<div class="hero"><span class="tag"><i></i>AI-powered health screening</span>'
        '<h1>Understand your diabetes risk. Act early.</h1>'
        '<p>Answer a few health questions to explore your estimated diabetes risk, with a clear explanation of what influenced it.</p>'
        '<svg class="ecg" viewBox="0 0 420 44" preserveAspectRatio="none"><path d="M0 24 H70 L82 24 L92 8 L104 40 '
        'L114 24 H190 L202 24 L212 6 L224 42 L234 24 H310 L322 24 L332 10 L344 38 L354 24 H420"/></svg></div>'
        '<div class="chips"><span class="chip c1">🩸 Diabetes screening</span><span class="chip c2">🌲 Random Forest model</span>'
        '<span class="chip c3">Educational tool · Not a medical diagnosis</span></div>',
        unsafe_allow_html=True,
    )
    if "quote" not in st.session_state:
        st.session_state["quote"] = random.choice(QUOTES)
    st.markdown(quote_html(*st.session_state["quote"]), unsafe_allow_html=True)
    st.write("")

    st.markdown('<div class="sec">⚡ Try an example</div>', unsafe_allow_html=True)
    e1, e2, e3 = st.columns([1, 1, 2])
    e1.button("Lower-risk example", on_click=load_example, args=("low",), key="ex_low")
    e2.button("Higher-risk example", on_click=load_example, args=("high",), key="ex_high")
    e3.markdown('<div class="help">Fills the form with sample values so you can see how the app works. '
                'The scores come from the trained model.</div>', unsafe_allow_html=True)
    st.write("")
    tab_basic, tab_more, tab_fam, tab_food, tab_model = st.tabs(
        ["①  Health details", "②  Optional measurements", "③  Family history", "④  Nutrition guide", "⑤  Model performance"])
    assumed = []

    with tab_basic:
        st.markdown('<div class="notice"><b>Who is this model for?</b>It was trained on health records of women aged 21 and '
                    'older (the Pima Indians Diabetes data format). Results for other groups may be less reliable.</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="sec">👤 Personal information</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        age = c1.number_input("Age (years, 21 or older)", 21, 90, key="age")
        preg = c2.number_input("Number of pregnancies", 0, 20, key="preg", help="Enter 0 if not applicable.")
        height = c3.number_input("Height (cm)", 120, 220, key="height")
        weight = c4.number_input("Weight (kg)", 30, 200, key="weight")
        bmi = round(weight / ((height / 100) ** 2), 1)
        st.markdown(f'<div class="bmi">Your BMI: {bmi} · {bmi_category(bmi)}</div>', unsafe_allow_html=True)

        st.markdown('<div class="sec">🩺 Health measurements</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        ogtt = c1.radio("Have you had a glucose tolerance test (OGTT)?", ["No or not sure", "Yes"], key="ogtt", horizontal=True,
                        help="A test where blood sugar is measured 2 hours after drinking a glucose solution.")
        if ogtt == "Yes":
            c1.number_input("Blood glucose 2 hours after the glucose drink (mg/dL)", 40, 300, key="glucose",
                            help="Use the 2-hour value from the test report.")
        else:
            c1.caption("A fasting or random sugar reading is a different test, so it is not used here. "
                       "A typical value will be used and your result will be marked.")
        with_unknown(c2, "Blood pressure: bottom number (mmHg)", 30, 140, "dbp",
                     help="Diastolic pressure, the lower number in a reading such as 120/80.")

    with tab_more:
        st.markdown('<div class="notice"><b>Optional measurements</b>These come from lab tests or a clinic visit. Most people '
                    'do not know them, so "I don\'t know" is ticked by default. Unknown values are replaced with the typical '
                    'value from the training data, and your results are clearly marked.</div>', unsafe_allow_html=True)
        a, b = st.columns(2)
        with_unknown(a, "Insulin, 2 hours after glucose drink (mu U/ml)", 0, 900, "insulin", help="Serum insulin from a lab test.")
        with_unknown(b, "Triceps skin fold thickness (mm)", 5, 100, "skin",
                     help="Measured with a skin-fold caliper at the back of the upper arm.")

    with tab_fam:
        st.markdown('<div class="notice"><b>Family history of diabetes</b>Diabetes in blood relatives is one of the inputs '
                    'the model uses. Add the relatives you know about, <b>including relatives who do not have diabetes</b>. '
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
            dpf = compute_dpf(rels)
            typ = float(MED["DiabetesPedigreeFunction"])
            word = "above" if dpf > typ * 1.1 else "below" if dpf < typ * 0.9 else "close to"
            st.markdown(f'<div class="bmi">Your family history score: {dpf:.2f} · {word} the typical value ({typ:.2f})</div>',
                        unsafe_allow_html=True)
        else:
            dpf = np.nan
            assumed.append("family history score")
            st.markdown('<div class="help">No relatives added yet. A typical family history score will be used.</div>',
                        unsafe_allow_html=True)
        with st.expander("How is the family history score calculated?"):
            st.markdown(
                "It follows the published *diabetes pedigree function* used to build this dataset (Smith et al., 1988):\n\n"
                "`score = (20 + sum of share × (88 − age at diagnosis) for relatives with diabetes) ÷ "
                "(50 + sum of share × (age − 14) for relatives without diabetes)`\n\n"
                "*Share* is the fraction of genes you have in common with the relative: 0.5 for parents and full siblings, "
                "0.25 for half siblings, grandparents, aunts and uncles, and 0.125 for half aunts or uncles and first cousins. "
                "Terms below zero count as zero. More affected close relatives raise the score, and healthy older relatives lower it.")

    with tab_food:
        st.markdown('<div class="help">What to eat and what to limit for a healthy blood sugar level. '
                    'This is general guidance, not a personal diet plan.</div>', unsafe_allow_html=True)
        st.markdown(plate_html(), unsafe_allow_html=True)
        cols = st.columns(3)
        for col, (cls, icon, name, items) in zip(cols, FOOD):
            with col:
                with st.expander(f"{icon} {name}", expanded=True):
                    st.markdown("\n".join(f"- {i}" for i in items))
        st.caption("Eat regular meals, watch portion sizes, drink plenty of water and stay active. If you are pregnant, "
                   "have kidney disease or take medicines for diabetes, follow your doctor's or dietitian's advice.")

    with tab_model:
        m = M["metrics"]
        tn, fp, fn, tp = M["cm"]
        st.markdown('<div class="help">How well the Random Forest performs on records it has never seen (a held-out 20% test set).</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="mgrid">' + "".join(
            f'<div class="mtile"><b>{v * 100:.0f}%</b><span>{k}</span></div>' for k, v in m.items()) + '</div>',
            unsafe_allow_html=True)
        st.markdown('<div class="sec">Confusion matrix (test set)</div>', unsafe_allow_html=True)
        st.markdown(
            '<table class="cm"><tr><th></th><th>Predicted: no diabetes</th><th>Predicted: diabetes</th></tr>'
            f'<tr><th>Actual: no diabetes</th><td><b>{tn}</b></td><td><b>{fp}</b></td></tr>'
            f'<tr><th>Actual: diabetes</th><td><b>{fn}</b></td><td><b>{tp}</b></td></tr></table>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="help"><b>Recall</b> shows how many people who really had diabetes were caught ({tp} of {tp + fn}). '
            f'<b>Precision</b> shows how many flagged people really had it ({tp} of {tp + fp}). For screening, recall matters '
            f'because missing a real case is worse than a false alarm.</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec">About the data and model</div>', unsafe_allow_html=True)
        st.markdown(
            f'- **Data:** {M["n"]} records, {M["pos_rate"] * 100:.0f}% with diabetes, 8 input features, {M["n_test"]} used for testing.\n'
            '- **Cleaning:** zero values in glucose, blood pressure, skin fold, insulin and BMI are treated as missing and '
            'filled with the training median.\n'
            '- **Model:** Random Forest with 200 trees, trained on 80% of the data (stratified split, random_state 42).\n'
            '- **Limits:** small dataset, one population group, and scores are not exact probabilities.')

    st.write("")
    st.button("Check my diabetes risk", type="primary", on_click=show_results, key="check")


st.markdown('<div class="foot"><b>HealthGuard AI</b> · Diabetes Risk Prediction · Random Forest · Streamlit · v1.1<br>'
            'Team:45 - Artificial Intelligence and Data Science · HNN College of Engineering<br>'
            'Educational screening tool, not a medical diagnosis. This app has no accounts or database and does not '
            'store your answers itself.</div>', unsafe_allow_html=True)
