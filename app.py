import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SpamGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(circle at 10% 10%, #172554 0%, transparent 30%),
        radial-gradient(circle at 90% 20%, #3b0764 0%, transparent 30%),
        linear-gradient(135deg, #020617, #0f172a);
    color: #f8fafc;
}

.main-title {
    font-size: 48px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 5px;
    background: linear-gradient(90deg, #22d3ee, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 18px;
    margin-bottom: 35px;
}

.card {
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 18px;
    padding: 25px;
    margin-bottom: 20px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.25);
}

.spam-result {
    background: rgba(127, 29, 29, 0.35);
    border: 1px solid #ef4444;
    border-radius: 18px;
    padding: 25px;
    text-align: center;
}

.ham-result {
    background: rgba(6, 78, 59, 0.35);
    border: 1px solid #22c55e;
    border-radius: 18px;
    padding: 25px;
    text-align: center;
}

.result-title {
    font-size: 30px;
    font-weight: 800;
}

.metric-card {
    background: rgba(30, 41, 59, 0.7);
    border-radius: 15px;
    padding: 18px;
    text-align: center;
    border: 1px solid rgba(148, 163, 184, 0.15);
}

textarea {
    background-color: #020617 !important;
    color: #e2e8f0 !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
}

.stButton > button {
    width: 100%;
    border-radius: 12px;
    border: none;
    padding: 12px;
    font-weight: 700;
    background: linear-gradient(90deg, #06b6d4, #8b5cf6);
    color: white;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(139,92,246,0.4);
}

section[data-testid="stSidebar"] {
    background: #020617;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🛡️ SpamGuard AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Intelligent Spam Detection using Machine Learning & NLP'
    '</div>',
    unsafe_allow_html=True
)

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    data = pd.read_csv(
        "spam.csv",
        encoding="latin-1"
    )

    # Standard SMS Spam Collection dataset
    if "v1" in data.columns and "v2" in data.columns:

        data = data[["v1", "v2"]]
        data.columns = ["label", "text"]

    else:
        # Support already-cleaned datasets
        if "label" not in data.columns or "text" not in data.columns:
            st.error(
                "Dataset must contain either "
                "'v1' and 'v2' or 'label' and 'text' columns."
            )
            st.stop()

        data = data[["label", "text"]]

    # Clean labels
    data["label"] = (
        data["label"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    data["label"] = data["label"].map({
        "ham": 0,
        "spam": 1
    })

    # Remove invalid rows
    data = data.dropna(subset=["label", "text"])

    # Convert text to string
    data["text"] = data["text"].astype(str).str.strip()

    # Remove empty messages
    data = data[data["text"] != ""]

    # Remove duplicate messages
    data = data.drop_duplicates(
        subset=["text"]
    )

    # Convert label to integer
    data["label"] = data["label"].astype(int)

    return data.reset_index(drop=True)


# ============================================================
# LOAD DATASET
# ============================================================

data = load_data()

# ============================================================
# DATASET STATISTICS
# ============================================================

total_messages = len(data)

spam_messages = int(
    (data["label"] == 1).sum()
)

ham_messages = int(
    (data["label"] == 0).sum()
)

spam_percentage = (
    spam_messages / total_messages * 100
)

ham_percentage = (
    ham_messages / total_messages * 100
)

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    data["text"],
    data["label"],
    test_size=0.20,
    random_state=42,
    stratify=data["label"]
)

# ============================================================
# CREATE MODELS
# ============================================================

def create_models():

    models = {

        "Naive Bayes": Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=1
                )
            ),
            (
                "model",
                MultinomialNB()
            )
        ]),

        "Logistic Regression": Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=1
                )
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            )
        ]),

        "Linear SVM": Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=1
                )
            ),
            (
                "model",
                LinearSVC(
                    random_state=42
                )
            )
        ])
    }

    return models


# ============================================================
# TRAIN MODELS
# ============================================================

@st.cache_resource
def train_models(
    X_train_data,
    X_test_data,
    y_train_data,
    y_test_data
):

    models = create_models()

    trained_models = {}
    results = {}
    predictions_store = {}

    for name, pipeline in models.items():

        pipeline.fit(
            X_train_data,
            y_train_data
        )

        predictions = pipeline.predict(
            X_test_data
        )

        accuracy = accuracy_score(
            y_test_data,
            predictions
        )

        precision = precision_score(
            y_test_data,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test_data,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test_data,
            predictions,
            zero_division=0
        )

        cm = confusion_matrix(
            y_test_data,
            predictions
        )

        results[name] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        }

        predictions_store[name] = {
            "predictions": predictions,
            "confusion_matrix": cm
        }

        trained_models[name] = pipeline

    return (
        trained_models,
        results,
        predictions_store
    )


trained_models, results, predictions_store = train_models(
    X_train,
    X_test,
    y_train,
    y_test
)

# ============================================================
# SELECT BEST MODEL
# ============================================================

best_model_name = max(
    results,
    key=lambda model_name:
        results[model_name]["F1 Score"]
)

best_model = trained_models[
    best_model_name
]

best_results = results[
    best_model_name
]

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🛡️ SpamGuard AI")

st.sidebar.markdown(
    "### Project Overview"
)

st.sidebar.write(
    "A machine learning and NLP system "
    "for detecting spam and legitimate messages."
)

st.sidebar.markdown("---")

st.sidebar.write(
    f"📊 **Dataset:** {total_messages:,} messages"
)

st.sidebar.write(
    f"🚫 **Spam:** {spam_messages:,}"
)

st.sidebar.write(
    f"✅ **Not Spam:** {ham_messages:,}"
)

st.sidebar.write(
    f"🏆 **Best Model:** {best_model_name}"
)

st.sidebar.markdown("---")

st.sidebar.write(
    f"📚 **Training Samples:** {len(X_train):,}"
)

st.sidebar.write(
    f"🧪 **Testing Samples:** {len(X_test):,}"
)

# ============================================================
# NAVIGATION
# ============================================================

page = st.sidebar.radio(
    "Navigate",
    [
        "🔍 Detector",
        "📊 Model Performance",
        "📈 Dataset Insights"
    ]
)

# ============================================================
# DETECTOR
# ============================================================

if page == "🔍 Detector":

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("📧 Analyze Your Message")

    st.write(
        "Enter an email or message below. "
        "The trained NLP system will classify it "
        "as Spam or Not Spam."
    )

    user_input = st.text_area(
        "Message Content",
        height=180,
        placeholder=(
            "Example: Congratulations! "
            "You have won a free prize..."
        )
    )

    if st.button("🔎 Analyze Message"):

        if not user_input.strip():

            st.warning(
                "⚠️ Please enter a message first."
            )

        else:

            prediction = best_model.predict(
                [user_input]
            )[0]

            # ----------------------------------------
            # PROBABILITY
            # ----------------------------------------

            spam_probability = None

            if hasattr(
                best_model,
                "predict_proba"
            ):

                probabilities = (
                    best_model
                    .predict_proba(
                        [user_input]
                    )[0]
                )

                spam_probability = probabilities[1]

            else:

                # LinearSVC does not provide
                # predict_proba.
                #
                # Convert decision score to
                # a confidence-like value.

                decision_score = (
                    best_model
                    .decision_function(
                        [user_input]
                    )[0]
                )

                confidence = (
                    1 /
                    (
                        1 +
                        np.exp(
                            -decision_score
                        )
                    )
                )

                spam_probability = confidence

            # ----------------------------------------
            # SPAM RESULT
            # ----------------------------------------

            if prediction == 1:

                st.markdown(
                    '<div class="spam-result">'
                    '<div class="result-title">'
                    '🚨 SPAM DETECTED'
                    '</div>'
                    '<p>'
                    'This message is classified as '
                    'potentially suspicious.'
                    '</p>'
                    '</div>',
                    unsafe_allow_html=True
                )

            # ----------------------------------------
            # HAM RESULT
            # ----------------------------------------

            else:

                st.markdown(
                    '<div class="ham-result">'
                    '<div class="result-title">'
                    '✅ NOT SPAM'
                    '</div>'
                    '<p>'
                    'This message appears to be '
                    'legitimate.'
                    '</p>'
                    '</div>',
                    unsafe_allow_html=True
                )

            # ----------------------------------------
            # PROBABILITY
            # ----------------------------------------

            st.markdown(
                "### 🎯 Spam Probability"
            )

            st.progress(
                float(
                    np.clip(
                        spam_probability,
                        0,
                        1
                    )
                )
            )

            st.write(
                f"**{spam_probability * 100:.2f}%**"
            )

            # ----------------------------------------
            # MODEL USED
            # ----------------------------------------

            st.info(
                f"🧠 Prediction generated using "
                f"**{best_model_name}** with TF-IDF features."
            )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

    # ========================================================
    # ML PIPELINE
    # ========================================================

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader(
        "🧠 Machine Learning Pipeline"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Feature Extraction",
            "TF-IDF"
        )

    with col2:

        st.metric(
            "Best Model",
            best_model_name
        )

    with col3:

        st.metric(
            "Classification",
            "Binary"
        )

    st.markdown(
        """
        **Pipeline**

        Raw Message → Text Preprocessing → TF-IDF
        → Machine Learning Model → Spam / Not Spam
        """
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📊 Model Performance":

    st.header(
        "📊 Model Performance"
    )

    st.write(
        "Three supervised machine learning "
        "algorithms are trained and evaluated "
        "using the same dataset split."
    )

    # ========================================================
    # BEST MODEL METRICS
    # ========================================================

    st.subheader(
        f"🏆 Best Model — {best_model_name}"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Accuracy",
            f"{best_results['Accuracy'] * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Precision",
            f"{best_results['Precision'] * 100:.2f}%"
        )

    with col3:

        st.metric(
            "Recall",
            f"{best_results['Recall'] * 100:.2f}%"
        )

    with col4:

        st.metric(
            "F1 Score",
            f"{best_results['F1 Score'] * 100:.2f}%"
        )

    st.markdown("---")

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    st.subheader(
        "🏆 Model Comparison"
    )

    comparison = pd.DataFrame(
        results
    ).T

    comparison_percent = (
        comparison * 100
    ).round(2)

    st.dataframe(
        comparison_percent,
        use_container_width=True
    )

    st.subheader(
        "📈 Performance Comparison"
    )

    st.bar_chart(
        comparison_percent[
            [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score"
            ]
        ]
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    st.subheader(
        "🔲 Confusion Matrix"
    )

    cm = predictions_store[
        best_model_name
    ]["confusion_matrix"]

    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual Not Spam",
            "Actual Spam"
        ],
        columns=[
            "Predicted Not Spam",
            "Predicted Spam"
        ]
    )

    st.dataframe(
        cm_df,
        use_container_width=True
    )

    st.caption(
        "The confusion matrix shows how many messages "
        "were correctly and incorrectly classified."
    )

# ============================================================
# DATASET INSIGHTS
# ============================================================

elif page == "📈 Dataset Insights":

    st.header(
        "📈 Dataset Insights"
    )

    # ========================================================
    # METRICS
    # ========================================================

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Messages",
            f"{total_messages:,}"
        )

    with col2:

        st.metric(
            "Spam Messages",
            f"{spam_messages:,}"
        )

    with col3:

        st.metric(
            "Legitimate Messages",
            f"{ham_messages:,}"
        )

    st.markdown("---")

    # ========================================================
    # CLASS DISTRIBUTION
    # ========================================================

    st.subheader(
        "📊 Class Distribution"
    )

    class_distribution = pd.DataFrame({
        "Category": [
            "Not Spam",
            "Spam"
        ],
        "Messages": [
            ham_messages,
            spam_messages
        ]
    })

    st.bar_chart(
        class_distribution.set_index(
            "Category"
        )
    )

    # ========================================================
    # CLASS PERCENTAGE
    # ========================================================

    st.subheader(
        "📌 Dataset Composition"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Not Spam",
            f"{ham_percentage:.2f}%"
        )

    with col2:

        st.metric(
            "Spam",
            f"{spam_percentage:.2f}%"
        )

    # ========================================================
    # DATASET PREVIEW
    # ========================================================

    st.subheader(
        "🔎 Dataset Preview"
    )

    preview = data.copy()

    preview["label"] = preview[
        "label"
    ].map({
        0: "Not Spam",
        1: "Spam"
    })

    st.dataframe(
        preview.head(20),
        use_container_width=True
    )

    # ========================================================
    # DATASET INFORMATION
    # ========================================================

    st.subheader(
        "ℹ️ Dataset Information"
    )

    st.write(
        f"""
        **Total unique messages:** {total_messages:,}

        **Spam messages:** {spam_messages:,}

        **Legitimate messages:** {ham_messages:,}

        **Training samples:** {len(X_train):,}

        **Testing samples:** {len(X_test):,}

        **Test split:** 20%

        **Feature extraction:** TF-IDF

        **Models evaluated:** 3
        """
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;color:#64748b;">
        <b>SpamGuard AI</b><br>
        Machine Learning + NLP • Intelligent Spam Detection
    </div>
    """,
    unsafe_allow_html=True
)