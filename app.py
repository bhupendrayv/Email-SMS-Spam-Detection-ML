"""
Streamlit Web Dashboard for Email & SMS Spam Detection, Phishing Analysis,
and Malicious URL Inspection.
Features:
- Real-time SMS & Email spam detection (Subject + Body)
- Phishing email detection (Text + indicators: urgency, link mismatches, credential harvesting)
- Malicious URL detection (Character n-gram TF-IDF + lexical features: length, dots, @, IP address, HTTPS)
- Spam probability percentage with confidence bar and explainable top influencing words
- Analytics Dashboard: total checked, spam %, daily trend line chart, pie chart, recent predictions
- Model Comparison Page: BoW vs TF-IDF benchmark charts, ROC curves, and confusion matrix leaderboard
"""

from __future__ import annotations

import datetime
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import SpamPredictor
from src.advanced_detectors import PhishingDetector, MaliciousURLDetector

REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

# Streamlit Page Config
st.set_page_config(
    page_title="AI Spam & Cyber Threat Detection Hub",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Session State Initialization for Live Analytics
# ─────────────────────────────────────────────────────────────────────────────
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = [
        {
            "timestamp": (datetime.datetime.now() - datetime.timedelta(days=4)).strftime("%Y-%m-%d %H:%M"),
            "date": (datetime.datetime.now() - datetime.timedelta(days=4)).strftime("%Y-%m-%d"),
            "channel": "SMS",
            "content": "Hey are we still meeting at 6pm for dinner?",
            "prediction": "Legitimate (Ham)",
            "spam_prob": 1.2,
            "confidence": 98.8,
        },
        {
            "timestamp": (datetime.datetime.now() - datetime.timedelta(days=3)).strftime("%Y-%m-%d %H:%M"),
            "date": (datetime.datetime.now() - datetime.timedelta(days=3)).strftime("%Y-%m-%d"),
            "channel": "Email",
            "content": "URGENT: Claim your £1000 Amazon gift card now! Call 08000930705",
            "prediction": "Spam",
            "spam_prob": 97.4,
            "confidence": 97.4,
        },
        {
            "timestamp": (datetime.datetime.now() - datetime.timedelta(days=2)).strftime("%Y-%m-%d %H:%M"),
            "date": (datetime.datetime.now() - datetime.timedelta(days=2)).strftime("%Y-%m-%d"),
            "channel": "Phishing",
            "content": "Action Required: Your PayPal account has been restricted. Update password immediately.",
            "prediction": "Spam",
            "spam_prob": 98.9,
            "confidence": 98.9,
        },
        {
            "timestamp": (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d %H:%M"),
            "date": (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d"),
            "channel": "URL",
            "content": "http://192.168.1.1/paypal-login-verify/auth.php",
            "prediction": "Spam",
            "spam_prob": 94.6,
            "confidence": 94.6,
        },
        {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "channel": "SMS",
            "content": "Can you review the attached quarterly slide deck?",
            "prediction": "Legitimate (Ham)",
            "spam_prob": 3.4,
            "confidence": 96.6,
        },
    ]


@st.cache_resource
def load_predictor(model_name: str, representation: str) -> SpamPredictor:
    """Load cached spam prediction engine."""
    return SpamPredictor(model_name=model_name, representation=representation)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Navigation & Settings
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=70)
st.sidebar.title("🛡️ Threat Detection Hub")

page = st.sidebar.radio(
    "Navigation Menu",
    [
        "⚡ Real-Time Threat Detector",
        "📊 Analytics Dashboard",
        "🔬 BoW vs TF-IDF Model Comparison",
        "📁 Batch Message Scanner",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Model Configuration")

model_choice = st.sidebar.selectbox(
    "Active Architecture",
    ["SVM (RBF Kernel)", "Linear SVM", "Logistic Regression", "Multinomial Naive Bayes", "Random Forest"],
    index=0,
)

model_key_map = {
    "SVM (RBF Kernel)": "svm_rbf",
    "Linear SVM": "linear_svm",
    "Logistic Regression": "logistic_regression",
    "Multinomial Naive Bayes": "multinomial_nb",
    "Random Forest": "random_forest",
}

rep_choice = st.sidebar.radio(
    "Text Representation",
    ["TFIDF", "BOW"],
    index=0,
    help="Compare Term Frequency-Inverse Document Frequency vs Bag-of-Words.",
)

# Load predictor
try:
    predictor = load_predictor(model_name=model_key_map[model_choice], representation=rep_choice.lower())
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    predictor = None


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1: REAL-TIME THREAT DETECTOR
# ─────────────────────────────────────────────────────────────────────────────
if page == "⚡ Real-Time Threat Detector":
    st.title("⚡ Real-Time Multi-Channel Threat Detector")
    st.markdown(
        "Detect and explain security threats across **SMS**, **Email (Subject + Body)**, "
        "**Phishing Communications**, and **Malicious URLs**."
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "📱 SMS Spam Detection",
        "✉️ Email Spam Detection",
        "🎣 Phishing Email Detection",
        "🌐 Malicious URL Detection",
    ])

    # 1. SMS Tab
    with tab1:
        st.subheader("📱 SMS Spam Analysis")
        sms_sample = st.selectbox(
            "Select a pre-loaded sample SMS:",
            [
                "Custom Text",
                "Spam: WINNER! You have won a £1000 prize or cash guarantee! Call 09061701461 to claim.",
                "Spam: Urgent! Your account statement shows unpaid invoice. Text STOP to cancel.",
                "Ham: Hey mate, are we still playing football tomorrow evening?",
                "Ham: I'll pick up groceries on my way home from work.",
            ],
        )

        if sms_sample == "Custom Text":
            sms_input = st.text_area("Enter SMS message content:", height=100, placeholder="Type or paste SMS here...")
        else:
            sample_val = sms_sample.split(": ", 1)[1] if ": " in sms_sample else sms_sample
            sms_input = st.text_area("Enter SMS message content:", value=sample_val, height=100)

        if st.button("🚀 Analyze SMS", key="btn_sms"):
            if not sms_input.strip():
                st.warning("Please enter an SMS message.")
            elif predictor is None:
                st.error("Predictor model not available.")
            else:
                res = predictor.predict(sms_input)
                is_spam = res["is_spam"]

                col_a, col_b = st.columns([1, 1.2])

                with col_a:
                    if is_spam:
                        st.error(f"### 🚨 Verdict: **SPAM DETECTED**")
                    else:
                        st.success(f"### ✅ Verdict: **LEGITIMATE (HAM)**")

                    st.metric("Spam Probability", f"{res['spam_probability_pct']}%")
                    st.progress(res["spam_probability_pct"] / 100.0)
                    st.caption(f"Model Confidence: **{res['confidence_pct']}%** using **{res['representation']} + {model_choice}**")

                with col_b:
                    st.markdown("#### 🔍 Explainability & Token Attribution")
                    triggers = res["top_influencing_words"]
                    if triggers:
                        df_trig = pd.DataFrame(triggers)
                        fig = px.bar(
                            df_trig,
                            x="score",
                            y="word",
                            orientation="h",
                            color="sentiment",
                            color_discrete_map={"Spam Trigger": "#e74c3c", "Ham Indicator": "#2ecc71"},
                            title="Influence of Extracted Words",
                        )
                        fig.update_layout(height=260, margin=dict(l=0, r=0, t=30, b=0))
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No high-leverage vocabulary tokens detected in feature dictionary.")

                # Record in session state
                st.session_state.prediction_history.append({
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "date": datetime.datetime.now().strftime("%Y-%m-%d"),
                    "channel": "SMS",
                    "content": sms_input[:60] + "..." if len(sms_input) > 60 else sms_input,
                    "prediction": "Spam" if is_spam else "Legitimate (Ham)",
                    "spam_prob": res["spam_probability_pct"],
                    "confidence": res["confidence_pct"],
                })

    # 2. Email Tab (Subject + Body)
    with tab2:
        st.subheader("✉️ Email Spam Detection (Subject + Body)")
        email_sub = st.text_input("Email Subject:", placeholder="e.g. Action Required: Invoice #8941 overdue")
        email_body = st.text_area(
            "Email Body:",
            height=140,
            placeholder="e.g. Dear customer, your monthly subscription payment failed. Please click below to verify payment...",
        )

        if st.button("🚀 Analyze Email", key="btn_email"):
            if not email_body.strip() and not email_sub.strip():
                st.warning("Please provide an email subject or body.")
            elif predictor is None:
                st.error("Predictor model not available.")
            else:
                res = predictor.predict_email(email_sub, email_body)
                is_spam = res["is_spam"]

                col_e1, col_e2 = st.columns([1, 1.2])
                with col_e1:
                    if is_spam:
                        st.error("### 🚨 Verdict: **SPAM EMAIL**")
                    else:
                        st.success("### ✅ Verdict: **LEGITIMATE EMAIL**")

                    st.metric("Spam Probability", f"{res['spam_probability_pct']}%")
                    st.progress(res["spam_probability_pct"] / 100.0)
                    st.caption(f"Architecture: **{res['representation']} + {model_choice}**")

                with col_e2:
                    st.markdown("#### 🔍 Influencing Email Keywords")
                    triggers = res["top_influencing_words"]
                    if triggers:
                        df_trig = pd.DataFrame(triggers)
                        fig = px.bar(
                            df_trig,
                            x="score",
                            y="word",
                            orientation="h",
                            color="sentiment",
                            color_discrete_map={"Spam Trigger": "#e74c3c", "Ham Indicator": "#2ecc71"},
                        )
                        fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0))
                        st.plotly_chart(fig, use_container_width=True)

                st.session_state.prediction_history.append({
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "date": datetime.datetime.now().strftime("%Y-%m-%d"),
                    "channel": "Email",
                    "content": f"Subj: {email_sub[:30]} | {email_body[:40]}",
                    "prediction": "Spam" if is_spam else "Legitimate (Ham)",
                    "spam_prob": res["spam_probability_pct"],
                    "confidence": res["confidence_pct"],
                })

    # 3. Phishing Tab
    with tab3:
        st.subheader("🎣 Phishing Email Detector (Heuristic + NLP Analysis)")
        st.markdown(
            "Detects social-engineering credential harvesting attacks by scanning for "
            "**urgency signals**, **credential requests**, **link mismatches**, and **suspicious CTAs**."
        )

        p_sub = st.text_input("Phishing Subject:", value="Urgent: Your Banking Password Expired - Reset Immediately")
        p_body = st.text_area(
            "Phishing Body:",
            value=(
                "Security Alert: We detected an unauthorized login attempt from an unknown device. "
                "Your account will be suspended within 24 hours unless you verify your password. "
                "Click here immediately: http://192.168.1.10/bankofamerica-login/auth.html"
            ),
            height=130,
        )

        if st.button("🔍 Scan for Phishing Attack", key="btn_phish"):
            phish_det = PhishingDetector(spam_detector=predictor)
            res = phish_det.predict(p_sub, p_body)
            indicators = res["indicators"]

            col_p1, col_p2 = st.columns([1, 1.3])
            with col_p1:
                if res["is_phishing"]:
                    st.error(f"### {res['verdict']}")
                else:
                    st.success(f"### {res['verdict']}")

                st.metric("Phishing Threat Probability", f"{res['phishing_probability']}%")
                st.progress(res["phishing_probability"] / 100.0)
                st.metric("Heuristic Risk Score", f"{res['threat_risk_score']} / 100")

            with col_p2:
                st.markdown("#### 🚨 Detected Phishing Indicators")
                st.write(f"• **Urgency Keywords**: {indicators['urgency_triggers'] or 'None'}")
                st.write(f"• **Credential Requests**: {indicators['credential_triggers'] or 'None'}")
                st.write(f"• **Suspicious CTAs**: {indicators['calls_to_action'] or 'None'}")
                st.write(f"• **IP Links / Mismatches**: {indicators['mismatched_suspicious_urls'] or 'None'}")

            st.session_state.prediction_history.append({
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "date": datetime.datetime.now().strftime("%Y-%m-%d"),
                "channel": "Phishing",
                "content": f"{p_sub[:40]}...",
                "prediction": "Spam" if res["is_phishing"] else "Legitimate (Ham)",
                "spam_prob": res["phishing_probability"],
                "confidence": res["phishing_probability"],
            })

    # 4. Malicious URL Tab
    with tab4:
        st.subheader("🌐 Malicious URL Detection")
        st.markdown(
            "Inspect URLs using **Character n-gram TF-IDF** combined with **Lexical structural heuristics**: "
            "URL length, dot count, `@` redirection symbols, raw IP hostnames, HTTPS presence, and brand spoof tokens."
        )

        url_input = st.text_input(
            "Enter URL to inspect:",
            value="http://paypal-security-update.com@account-check.xyz/signin/login.php",
            placeholder="https://example.com/login",
        )

        if st.button("🛡️ Inspect URL Security", key="btn_url"):
            url_det = MaliciousURLDetector()
            res = url_det.predict(url_input)
            lex = res["lexical_features"]

            col_u1, col_u2 = st.columns([1, 1.3])
            with col_u1:
                if res["is_malicious"]:
                    st.error(f"### {res['verdict']}")
                else:
                    st.success(f"### {res['verdict']}")

                st.metric("Malicious Threat Probability", f"{res['threat_probability']}%")
                st.progress(res["threat_probability"] / 100.0)
                st.metric("Lexical Risk Score", f"{lex['heuristic_risk_score']} / 100")

            with col_u2:
                st.markdown("#### 🔎 Lexical Feature Breakdown")
                u_df = pd.DataFrame([
                    {"Feature": "URL Length", "Value": lex["url_length"], "Risk": "⚠️ High" if lex["url_length"] > 75 else "Normal"},
                    {"Feature": "Subdomain Dots", "Value": lex["num_dots"], "Risk": "⚠️ High" if lex["num_dots"] > 3 else "Normal"},
                    {"Feature": "@ Redirection Symbol", "Value": str(lex["has_at_symbol"]), "Risk": "🚨 Severe" if lex["has_at_symbol"] else "Safe"},
                    {"Feature": "Raw IP Address Host", "Value": str(lex["has_ip_address"]), "Risk": "🚨 Severe" if lex["has_ip_address"] else "Safe"},
                    {"Feature": "HTTPS Secured", "Value": str(lex["has_https"]), "Risk": "Safe" if lex["has_https"] else "⚠️ Unencrypted"},
                    {"Feature": "Suspicious Keywords", "Value": ", ".join(lex["suspicious_keywords"]) or "None", "Risk": "⚠️ Detected" if lex["suspicious_keywords"] else "None"},
                ])
                st.dataframe(u_df, use_container_width=True, hide_index=True)

            st.session_state.prediction_history.append({
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "date": datetime.datetime.now().strftime("%Y-%m-%d"),
                "channel": "URL",
                "content": url_input[:50],
                "prediction": "Spam" if res["is_malicious"] else "Legitimate (Ham)",
                "spam_prob": res["threat_probability"],
                "confidence": res["threat_probability"],
            })


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: ANALYTICS DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Analytics Dashboard":
    st.title("📊 Security Operations Center (SOC) Dashboard")
    st.markdown("Live operational metrics, traffic distribution, and threat trajectory.")

    history_df = pd.DataFrame(st.session_state.prediction_history)
    total_checked = len(history_df)
    spam_count = len(history_df[history_df["prediction"] == "Spam"])
    ham_count = total_checked - spam_count
    spam_pct = (spam_count / max(total_checked, 1)) * 100.0
    avg_conf = history_df["confidence"].mean() if not history_df.empty else 0.0

    # Top KPI Metrics
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Inbound Inspected", f"{total_checked:,}", delta="+12% today")
    kpi2.metric("Spam / Threats Flagged", f"{spam_count}", delta=f"{spam_pct:.1f}% rate", delta_color="inverse")
    kpi3.metric("Legitimate (Ham)", f"{ham_count}")
    kpi4.metric("Avg Detection Confidence", f"{avg_conf:.1f}%")

    st.markdown("---")

    col_chart1, col_chart2 = st.columns([1.5, 1])

    with col_chart1:
        st.subheader("📈 Threat Volume Trajectory")
        if not history_df.empty and "date" in history_df.columns:
            daily_trends = history_df.groupby(["date", "prediction"]).size().reset_index(name="count")
            fig_trend = px.line(
                daily_trends,
                x="date",
                y="count",
                color="prediction",
                markers=True,
                color_discrete_map={"Spam": "#e74c3c", "Legitimate (Ham)": "#2ecc71"},
                title="Inspections Over Time by Class",
            )
            fig_trend.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("No threat inspection history recorded yet.")

    with col_chart2:
        st.subheader("🥧 Threat Class Ratio")
        if total_checked > 0:
            fig_pie = px.pie(
                values=[spam_count, ham_count],
                names=["Spam / Malicious", "Legitimate (Ham)"],
                color=["Spam / Malicious", "Legitimate (Ham)"],
                color_discrete_map={"Spam / Malicious": "#e74c3c", "Legitimate (Ham)": "#2ecc71"},
                hole=0.45,
            )
            fig_pie.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No threat class data available.")

    st.subheader("🕒 Recent Threat Log")
    st.dataframe(
        history_df.sort_values(by="timestamp", ascending=False),
        use_container_width=True,
        hide_index=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3: MODEL COMPARISON (BoW vs TF-IDF)
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔬 BoW vs TF-IDF Model Comparison":
    st.title("🔬 Core Innovation: Bag-of-Words vs. TF-IDF Benchmark")
    st.markdown(
        "Formal empirical comparison between **Bag-of-Words (CountVectorizer)** and "
        "**TF-IDF (TfidfVectorizer)** across 5 machine learning architectures."
    )

    eval_csv_path = REPORTS_DIR / "evaluation_comparison_table.csv"
    if eval_csv_path.exists():
        eval_df = pd.read_csv(eval_csv_path)

        # Leaderboard Table
        st.subheader("🏆 Test Set Benchmark Leaderboard")
        st.dataframe(
            eval_df[
                [
                    "representation",
                    "model_name",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1_score",
                    "roc_auc",
                    "latency_per_sample_ms",
                ]
            ].sort_values(by="f1_score", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("---")

        # Interactive Plotly Comparison Bar Chart
        st.subheader("📊 Interactive Metric Comparison: BoW vs TF-IDF")
        metric_to_plot = st.selectbox(
            "Select Evaluation Metric to Compare:",
            ["f1_score", "precision", "recall", "accuracy", "roc_auc"],
            index=0,
        )

        fig_bar = px.bar(
            eval_df,
            x="model_name",
            y=metric_to_plot,
            color="representation",
            barmode="group",
            color_discrete_map={"BOW": "#3498db", "TFIDF": "#e67e22"},
            title=f"Comparative {metric_to_plot.upper()} by Model Architecture",
            text_auto=".3f",
        )
        fig_bar.update_layout(height=450, yaxis_range=[0.75, 1.02])
        st.plotly_chart(fig_bar, use_container_width=True)

        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.subheader("📈 Comparative ROC Curves")
            roc_img = REPORTS_DIR / "roc_curves_comparison.png"
            if roc_img.exists():
                st.image(str(roc_img), caption="BoW vs TF-IDF ROC Curves (Test Split)")
        with col_img2:
            st.subheader("📊 Multi-Metric Comparison Grid")
            bar_img = REPORTS_DIR / "bow_vs_tfidf_comparison_bars.png"
            if bar_img.exists():
                st.image(str(bar_img), caption="Accuracy, Precision, Recall, and F1 Grid")

        # Final recommendation display
        rec_report = REPORTS_DIR / "model_recommendation_report.md"
        if rec_report.exists():
            with open(rec_report, "r", encoding="utf-8") as f:
                rec_text = f.read()
            with st.expander("📄 View Full Analytical Recommendation Report"):
                st.markdown(rec_text)
    else:
        st.info("Evaluation results not found. Run python src/evaluate.py to generate benchmarks.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4: BATCH MESSAGE SCANNER
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📁 Batch Message Scanner":
    st.title("📁 Bulk Batch Message Scanner")
    st.markdown("Upload a CSV file with a `text` column or paste multiple messages to classify in bulk.")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(batch_df)} rows.")

            text_col = st.selectbox("Select message text column:", batch_df.columns)

            if st.button("🚀 Classify Batch"):
                if predictor is None:
                    st.error("Predictor model not available.")
                else:
                    with st.spinner("Processing batch messages..."):
                        preds = []
                        probs = []
                        for t in batch_df[text_col]:
                            r = predictor.predict(str(t))
                            preds.append(r["prediction"])
                            probs.append(r["spam_probability_pct"])

                        batch_df["Prediction"] = preds
                        batch_df["Spam_Probability_%"] = probs

                    st.success("Batch classification complete!")
                    st.dataframe(batch_df.head(50), use_container_width=True)

                    csv_bytes = batch_df.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Download Classified Results CSV", data=csv_bytes, file_name="classified_results.csv")
        except Exception as e:
            st.error(f"Error parsing uploaded file: {e}")
