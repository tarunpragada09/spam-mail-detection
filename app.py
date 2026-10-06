import io
import re
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="SpamGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/228/"
    "sms+spam+collection.zip"
)

DATASET_FILE = Path("spam.csv")


# =========================================================
# PREMIUM CSS
# =========================================================

st.markdown(
    """
<style>

.stApp {
    background:
        radial-gradient(
            circle at 10% 0%,
            rgba(37,99,235,.18),
            transparent 28%
        ),
        radial-gradient(
            circle at 90% 10%,
            rgba(124,58,237,.16),
            transparent 25%
        ),
        #050816;
    color: #e5e7eb;
}

section[data-testid="stSidebar"] {
    background: #080d1c;
    border-right: 1px solid rgba(148,163,184,.12);
}

.hero {
    padding: 25px 0 15px 0;
}

.eyebrow {
    color: #60a5fa;
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 2px;
}

.hero-title {
    font-size: 64px;
    line-height: 1;
    font-weight: 900;
    letter-spacing: -3px;
    color: #f8fafc;
    margin-top: 8px;
}

.hero-text {
    color: #94a3b8;
    font-size: 17px;
    max-width: 800px;
}

.glass {
    padding: 22px;
    border-radius: 20px;
    background: rgba(15,23,42,.72);
    border: 1px solid rgba(148,163,184,.13);
    box-shadow: 0 18px 60px rgba(0,0,0,.25);
}

.risk-card {
    padding: 25px;
    border-radius: 20px;
    border: 1px solid rgba(239,68,68,.55);
    background:
        linear-gradient(
            135deg,
            rgba(127,29,29,.70),
            rgba(30,41,59,.85)
        );
    box-shadow: 0 0 40px rgba(239,68,68,.20);
}

.safe-card {
    padding: 25px;
    border-radius: 20px;
    border: 1px solid rgba(34,197,94,.45);
    background:
        linear-gradient(
            135deg,
            rgba(20,83,45,.60),
            rgba(15,23,42,.85)
        );
    box-shadow: 0 0 35px rgba(34,197,94,.12);
}

.small-label {
    color: #94a3b8;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 800;
}

.big-number {
    color: #f8fafc;
    font-size: 30px;
    font-weight: 900;
}

.footer {
    text-align: center;
    color: #64748b;
    padding: 30px 0 10px;
}

div.stButton > button {
    min-height: 46px;
    border-radius: 12px;
    font-weight: 800;
}

</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# DOWNLOAD / PREPARE DATASET
# =========================================================

@st.cache_data(show_spinner=False)
def prepare_dataset():

    # Check existing CSV
    if DATASET_FILE.exists():

        try:
            old_df = pd.read_csv(
                DATASET_FILE,
                encoding="latin-1"
            )

            # If already a proper large dataset,
            # use it.
            if len(old_df) >= 1000:
                return old_df

        except Exception:
            pass

    # Download official UCI dataset
    request = Request(
        DATASET_URL,
        headers={
            "User-Agent": "SpamGuardAI/1.0"
        }
    )

    with urlopen(
        request,
        timeout=30
    ) as response:

        data = response.read()

    with zipfile.ZipFile(
        io.BytesIO(data)
    ) as archive:

        target = None

        for name in archive.namelist():

            if name.lower().endswith(
                "smsspamcollection"
            ):
                target = name
                break

        if target is None:
            raise RuntimeError(
                "SMS Spam Collection file not found."
            )

        raw = archive.read(
            target
        ).decode(
            "utf-8"
        )

    rows = []

    for line in raw.splitlines():

        parts = line.split(
            "\t",
            1
        )

        if len(parts) == 2:

            label = parts[0].strip().lower()
            message = parts[1].strip()

            if label in ["ham", "spam"]:

                rows.append(
                    [label, message]
                )

    df = pd.DataFrame(
        rows,
        columns=[
            "v1",
            "v2"
        ]
    )

    df = df.dropna()

    df = df.drop_duplicates(
        subset=["v2"]
    )

    df.to_csv(
        DATASET_FILE,
        index=False,
        encoding="utf-8"
    )

    return df


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data(show_spinner=False)
def load_data():

    df = prepare_dataset()

    df = df[
        ["v1", "v2"]
    ].copy()

    df.columns = [
        "label",
        "message"
    ]

    df["label"] = (
        df["label"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df["message"] = (
        df["message"]
        .astype(str)
        .str.strip()
    )

    df = df[
        df["label"].isin(
            ["ham", "spam"]
        )
    ]

    df = df[
        df["message"] != ""
    ]

    df = df.drop_duplicates(
        subset=["message"]
    )

    df["target"] = (
        df["label"]
        .map(
            {
                "ham": 0,
                "spam": 1
            }
        )
    )

    # Additional dataset features
    df["char_count"] = (
        df["message"].str.len()
    )

    df["word_count"] = (
        df["message"]
        .str.split()
        .str.len()
    )

    df["url_count"] = (
        df["message"]
        .str.count(
            r"(https?://\S+|www\.\S+)"
        )
    )

    df["digit_count"] = (
        df["message"]
        .str.count(r"\d")
    )

    df["exclamation_count"] = (
        df["message"]
        .str.count("!")
    )

    return df.reset_index(
        drop=True
    )


# =========================================================
# TF-IDF
# =========================================================

def create_vectorizer():

    return TfidfVectorizer(

        lowercase=True,

        strip_accents="unicode",

        sublinear_tf=True,

        ngram_range=(1, 2),

        min_df=2,

        max_df=0.98,

        max_features=30000,

        token_pattern=(
            r"(?u)\b\w[\w$£€]*\b"
        )
    )


# =========================================================
# TRAIN MODELS
# =========================================================

@st.cache_resource(show_spinner=False)
def train_models(df):

    X = df["message"]

    y = df["target"]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )

    models = {

        "Naive Bayes":

        Pipeline(
            [
                (
                    "tfidf",
                    create_vectorizer()
                ),

                (
                    "classifier",
                    MultinomialNB(
                        alpha=0.25
                    )
                )
            ]
        ),

        "Logistic Regression":

        Pipeline(
            [
                (
                    "tfidf",
                    create_vectorizer()
                ),

                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1500,
                        class_weight="balanced",
                        C=2.0
                    )
                )
            ]
        ),

        "Calibrated Linear SVM":

        Pipeline(
            [
                (
                    "tfidf",
                    create_vectorizer()
                ),

                (
                    "classifier",
                    CalibratedClassifierCV(
                        estimator=LinearSVC(
                            C=1.5,
                            class_weight="balanced"
                        ),
                        method="sigmoid",
                        cv=5
                    )
                )
            ]
        )
    }

    results = {}

    for name, model in models.items():

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        probabilities = (
            model.predict_proba(
                X_test
            )[:, 1]
        )

        results[name] = {

            "model": model,

            "accuracy":
                accuracy_score(
                    y_test,
                    predictions
                ),

            "precision":
                precision_score(
                    y_test,
                    predictions,
                    zero_division=0
                ),

            "recall":
                recall_score(
                    y_test,
                    predictions,
                    zero_division=0
                ),

            "f1":
                f1_score(
                    y_test,
                    predictions,
                    zero_division=0
                ),

            "confusion":
                confusion_matrix(
                    y_test,
                    predictions
                )
        }

    best_model_name = max(
        results,
        key=lambda name:
        results[name]["f1"]
    )

    return (
        results,
        best_model_name,
        len(X_train),
        len(X_test)
    )


# =========================================================
# SUSPICIOUS INDICATORS
# =========================================================

def detect_indicators(message):

    indicators = {

        "Urgency":
        bool(
            re.search(
                r"\b("
                r"urgent|"
                r"immediately|"
                r"act now|"
                r"limited time|"
                r"last chance|"
                r"claim now"
                r")\b",
                message,
                re.I
            )
        ),

        "URL":
        bool(
            re.search(
                r"(https?://\S+|www\.\S+)",
                message,
                re.I
            )
        ),

        "Prize / Money":
        bool(
            re.search(
                r"\b("
                r"win|winner|won|"
                r"prize|cash|reward|"
                r"free|lottery|offer"
                r")\b",
                message,
                re.I
            )
        ),

        "OTP / Credentials":
        bool(
            re.search(
                r"\b("
                r"otp|password|pin|"
                r"bank|account|"
                r"verify|verification|"
                r"login"
                r")\b",
                message,
                re.I
            )
        ),

        "Phone Number":
        bool(
            re.search(
                r"\+?\d[\d\s-]{7,}\d",
                message
            )
        )
    }

    return indicators


def risk_level(probability):

    if probability >= 0.85:
        return "CRITICAL"

    if probability >= 0.65:
        return "HIGH"

    if probability >= 0.40:
        return "MEDIUM"

    return "LOW"


# =========================================================
# START
# =========================================================

with st.spinner(
    "🧠 Loading dataset and training AI models..."
):

    try:

        df = load_data()

        (
            results,
            best_model_name,
            train_count,
            test_count
        ) = train_models(df)

    except Exception as error:

        st.error(
            "❌ Could not load the dataset "
            "or train the models."
        )

        st.exception(error)

        st.stop()


best_model = (
    results[
        best_model_name
    ]["model"]
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        "## 🛡️ SpamGuard AI"
    )

    st.caption(
        "AI-powered spam detection"
    )

    st.divider()

    st.markdown(
        "### 📊 Dataset"
    )

    st.metric(
        "Total Messages",
        f"{len(df):,}"
    )

    st.metric(
        "Spam",
        f"{int(df['target'].sum()):,}"
    )

    st.metric(
        "Not Spam",
        f"{int((df['target'] == 0).sum()):,}"
    )

    st.divider()

    st.markdown(
        "### 🏆 Best Model"
    )

    st.success(
        best_model_name
    )

    st.caption(
        f"Training: {train_count:,}"
    )

    st.caption(
        f"Testing: {test_count:,}"
    )

    st.divider()

    st.caption(
        "Dataset: UCI SMS Spam Collection"
    )


# =========================================================
# HERO
# =========================================================

st.markdown(
    """
<div class="hero">

<div class="eyebrow">
AI • NLP • CYBERSECURITY
</div>

<div class="hero-title">
🛡️ SpamGuard AI
</div>

<div class="hero-text">
Intelligent spam detection using TF-IDF,
supervised machine learning and calibrated
confidence scoring.
</div>

</div>
""",
    unsafe_allow_html=True
)


# =========================================================
# METRICS
# =========================================================

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Messages",
    f"{len(df):,}"
)

c2.metric(
    "Spam Rate",
    f"{df['target'].mean() * 100:.1f}%"
)

c3.metric(
    "Best F1",
    f"{results[best_model_name]['f1'] * 100:.2f}%"
)

c4.metric(
    "Models",
    "3"
)


st.divider()


# =========================================================
# NAVIGATION
# =========================================================

page = st.radio(

    "Workspace",

    [
        "📧 Detector",
        "📈 Model Lab",
        "📊 Dataset Intelligence"
    ],

    horizontal=True,

    label_visibility="collapsed"
)


# =========================================================
# DETECTOR
# =========================================================

if page == "📧 Detector":

    left, right = st.columns(
        [1.45, 0.75],
        gap="large"
    )

    with left:

        st.subheader(
            "📧 Spam Message Detector"
        )

        st.caption(
            "Paste an SMS or email message "
            "and let the AI analyze it."
        )

        message = st.text_area(

            "Message",

            height=220,

            placeholder=(
                "Paste a suspicious message here..."
            ),

            label_visibility="collapsed"
        )

        analyze = st.button(

            "🔎 ANALYZE MESSAGE",

            type="primary",

            use_container_width=True
        )

        if analyze:

            if not message.strip():

                st.warning(
                    "⚠️ Please enter a message."
                )

            else:

                prediction = int(
                    best_model.predict(
                        [message]
                    )[0]
                )

                probability = float(
                    best_model.predict_proba(
                        [message]
                    )[0][1]
                )

                risk = risk_level(
                    probability
                )

                indicators = (
                    detect_indicators(
                        message
                    )
                )

                detected = [
                    name
                    for name, value
                    in indicators.items()
                    if value
                ]

                st.divider()

                if prediction == 1:

                    st.markdown(
                        f"""
<div class="risk-card">

<div class="small-label">
THREAT DETECTED
</div>

<h1>
🚨 SPAM MESSAGE
</h1>

<p>
The AI model classified this message
as likely spam.
</p>

<h2>
{probability * 100:.2f}%
</h2>

<p>
<b>Spam confidence</b>
</p>

<p>
<b>Risk Level:</b> {risk}
</p>

</div>
""",
                        unsafe_allow_html=True
                    )

                else:

                    ham_confidence = (
                        1 - probability
                    )

                    st.markdown(
                        f"""
<div class="safe-card">

<div class="small-label">
ANALYSIS COMPLETE
</div>

<h1>
✅ LIKELY NOT SPAM
</h1>

<p>
No strong spam pattern was detected.
</p>

<h2>
{ham_confidence * 100:.2f}%
</h2>

<p>
<b>Ham confidence</b>
</p>

<p>
<b>Risk Level:</b> {risk}
</p>

</div>
""",
                        unsafe_allow_html=True
                    )

                st.progress(

                    min(
                        max(
                            probability,
                            0.0
                        ),
                        1.0
                    ),

                    text=(
                        f"Spam probability: "
                        f"{probability * 100:.2f}%"
                    )
                )

                st.subheader(
                    "🔍 Suspicious Indicators"
                )

                if detected:

                    cols = st.columns(
                        min(
                            len(detected),
                            3
                        )
                    )

                    for i, item in enumerate(
                        detected
                    ):

                        cols[
                            i % len(cols)
                        ].warning(
                            f"⚠️ {item}"
                        )

                else:

                    st.success(
                        "No obvious suspicious "
                        "indicators detected."
                    )

                if prediction == 1:

                    st.warning(
                        "🚫 Security reminder: "
                        "Never share OTPs, passwords, "
                        "banking credentials or personal "
                        "information with unknown senders."
                    )

    with right:

        st.subheader(
            "⚙️ AI Model Status"
        )

        st.markdown(
            f"""
<div class="glass">

<div class="small-label">
ACTIVE MODEL
</div>

<div class="big-number">
{best_model_name}
</div>

<br>

<div class="small-label">
F1 SCORE
</div>

<div class="big-number">
{results[best_model_name]['f1'] * 100:.2f}%
</div>

<br>

<div class="small-label">
EVALUATION
</div>

<div>
Stratified 80/20 holdout
</div>

</div>
""",
            unsafe_allow_html=True
        )

        st.subheader(
            "🧠 ML Pipeline"
        )

        steps = [

            "1. Dataset validation",

            "2. Duplicate removal",

            "3. TF-IDF features",

            "4. Word + bigram features",

            "5. Multiple ML models",

            "6. Probability calibration",

            "7. F1-based model selection"
        ]

        for step in steps:

            st.write(
                f"✓ {step}"
            )


# =========================================================
# MODEL LAB
# =========================================================

elif page == "📈 Model Lab":

    st.header(
        "📈 Model Performance Lab"
    )

    performance = []

    for name, result in results.items():

        performance.append(

            {

                "Model": name,

                "Accuracy":
                    result["accuracy"] * 100,

                "Precision":
                    result["precision"] * 100,

                "Recall":
                    result["recall"] * 100,

                "F1 Score":
                    result["f1"] * 100
            }
        )

    performance_df = pd.DataFrame(
        performance
    )

    st.dataframe(

        performance_df.style.format(

            {

                "Accuracy":
                    "{:.2f}%",

                "Precision":
                    "{:.2f}%",

                "Recall":
                    "{:.2f}%",

                "F1 Score":
                    "{:.2f}%"
            }
        ),

        use_container_width=True,

        hide_index=True
    )

    st.subheader(
        f"🏆 Selected Model: "
        f"{best_model_name}"
    )

    best = results[
        best_model_name
    ]

    a, b, c, d = st.columns(4)

    a.metric(
        "Accuracy",
        f"{best['accuracy'] * 100:.2f}%"
    )

    b.metric(
        "Precision",
        f"{best['precision'] * 100:.2f}%"
    )

    c.metric(
        "Recall",
        f"{best['recall'] * 100:.2f}%"
    )

    d.metric(
        "F1",
        f"{best['f1'] * 100:.2f}%"
    )

    st.subheader(
        "Confusion Matrix"
    )

    matrix = pd.DataFrame(

        best["confusion"],

        index=[
            "Actual Ham",
            "Actual Spam"
        ],

        columns=[
            "Predicted Ham",
            "Predicted Spam"
        ]
    )

    st.dataframe(
        matrix,
        use_container_width=True
    )

    st.info(
        "The Linear SVM uses probability calibration "
        "with cross-validation instead of treating its "
        "raw decision score as a probability."
    )


# =========================================================
# DATASET INTELLIGENCE
# =========================================================

else:

    st.header(
        "📊 Dataset Intelligence"
    )

    a, b = st.columns(2)

    with a:

        st.subheader(
            "Message Distribution"
        )

        distribution = (
            df["label"]
            .value_counts()
        )

        st.bar_chart(
            distribution
        )

    with b:

        st.subheader(
            "Average Message Length"
        )

        length_data = (
            df.groupby(
                "label"
            )["char_count"]
            .mean()
            .round(1)
        )

        st.bar_chart(
            length_data
        )

    st.subheader(
        "🔬 Feature Analysis"
    )

    feature_table = pd.DataFrame(

        {

            "Feature": [

                "Average Characters",

                "Average Words",

                "Average URLs",

                "Average Digits",

                "Average Exclamations"
            ],

            "Ham": [

                df.loc[
                    df.target == 0,
                    "char_count"
                ].mean(),

                df.loc[
                    df.target == 0,
                    "word_count"
                ].mean(),

                df.loc[
                    df.target == 0,
                    "url_count"
                ].mean(),

                df.loc[
                    df.target == 0,
                    "digit_count"
                ].mean(),

                df.loc[
                    df.target == 0,
                    "exclamation_count"
                ].mean()
            ],

            "Spam": [

                df.loc[
                    df.target == 1,
                    "char_count"
                ].mean(),

                df.loc[
                    df.target == 1,
                    "word_count"
                ].mean(),

                df.loc[
                    df.target == 1,
                    "url_count"
                ].mean(),

                df.loc[
                    df.target == 1,
                    "digit_count"
                ].mean(),

                df.loc[
                    df.target == 1,
                    "exclamation_count"
                ].mean()
            ]
        }
    )

    st.dataframe(

        feature_table.style.format(

            {
                "Ham": "{:.2f}",
                "Spam": "{:.2f}"
            }
        ),

        use_container_width=True,

        hide_index=True
    )

    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(

        df[
            [
                "label",
                "message"
            ]
        ].head(25),

        use_container_width=True,

        hide_index=True
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
<div class="footer">
SpamGuard AI • Machine Learning + NLP
</div>
""",
    unsafe_allow_html=True
)