# =========================================
# 📦 IMPORTS
# =========================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc

# =========================================
# ⚙️ PAGE CONFIG
# =========================================
st.set_page_config(page_title="Placement Dashboard", layout="wide")
st.title("📊 Job Acceptance Prediction Dashboard")
st.markdown("---")

# =========================================
# 🚀 CACHE
# =========================================
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_dataset.csv")
    df.columns = df.columns.str.strip().str.lower()
    return df

@st.cache_resource
def load_model():
    try:
        model = pickle.load(open("model.pkl", "rb"))
        scaler = pickle.load(open("scaler.pkl", "rb"))
        return model, scaler
    except:
        return None, None

df = load_data()
model, scaler = load_model()

# =========================================
# 🎛️ FILTERS
# =========================================
st.sidebar.header("🔍 Filters")

exp_range = st.sidebar.slider("Experience", 0, 10, (0, 10))
skills_range = st.sidebar.slider("Skills %", 0, 100, (40, 100))

filtered_df = df[
    (df['years_of_experience'] >= exp_range[0]) &
    (df['years_of_experience'] <= exp_range[1]) &
    (df['skills_match_percentage'] >= skills_range[0]) &
    (df['skills_match_percentage'] <= skills_range[1])
]

# =========================================
# 📊 KPI
# =========================================
total_candidates = len(filtered_df)
placement_rate = filtered_df['status'].mean() * 100
job_acceptance_rate = placement_rate

avg_interview_score = (
    filtered_df[['technical_score', 'aptitude_score', 'communication_score']]
    .mean(axis=1).mean()
)

avg_skills_match = filtered_df['skills_match_percentage'].mean()
offer_dropout_rate = 100 - job_acceptance_rate

high_risk = filtered_df[
    (filtered_df['skills_match_percentage'] < 60) |
    (filtered_df['employment_gap_months'] > 12)
]

high_risk_percentage = (len(high_risk) / total_candidates) * 100 if total_candidates > 0 else 0

# =========================================
# 📈 KPI DISPLAY
# =========================================
col1, col2, col3 = st.columns(3)
col1.metric("Total Candidates", total_candidates)
col2.metric("Placement Rate (%)", f"{placement_rate:.2f}%")
col3.metric("Job Acceptance Rate (%)", f"{job_acceptance_rate:.2f}%")

col4, col5, col6 = st.columns(3)
col4.metric("Avg Interview Score", f"{avg_interview_score:.2f}")
col5.metric("Avg Skills Match (%)", f"{avg_skills_match:.2f}%")
col6.metric("Offer Dropout Rate (%)", f"{offer_dropout_rate:.2f}%")

col7, _ = st.columns(2)
col7.metric("High-Risk Candidate (%)", f"{high_risk_percentage:.2f}%")

st.markdown("---")

# =========================================
# 📊 TABS
# =========================================
tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Trends", "🎯 Prediction"])

# =========================================
# 📊 OVERVIEW
# =========================================
with tab1:

    placement_counts = filtered_df['status'].value_counts().sort_index()
    placement_counts.index = ["Not Placed", "Placed"]

    st.subheader("Placement Distribution")

    fig1, ax1 = plt.subplots()
    ax1.bar(placement_counts.index, placement_counts.values,
            color=['#ef5350', '#66bb6a'])  # soft red, green
    st.pyplot(fig1)

    st.subheader("Placement Share")

    fig2, ax2 = plt.subplots()
    ax2.pie(placement_counts,
            labels=["Not Placed", "Placed"],
            colors=['#ef5350', '#66bb6a'],
            autopct='%1.1f%%',
            startangle=90)
    st.pyplot(fig2)

# =========================================
# 📈 TRENDS
# =========================================
with tab2:
    #1️⃣Skills vs Placement
    st.subheader("Skills vs Placement")

    skills_plot = filtered_df.groupby('skills_match_percentage')['status'].mean()
    fig3, ax3 = plt.subplots()
    ax3.plot(skills_plot, color='#42a5f5')  # blue
    st.pyplot(fig3)

    #2️⃣Experience vs Placement
    st.subheader("Experience vs Placement")

    exp_plot = filtered_df.groupby('years_of_experience')['status'].mean()
    fig4, ax4 = plt.subplots()
    ax4.plot(exp_plot, color='#ab47bc')  # purple
    st.pyplot(fig4)
    # 3️⃣ Interview Scores Distribution
    st.subheader("📉 Interview Scores Comparison")
    fig_int, ax_int = plt.subplots()
    avg_scores = filtered_df[['technical_score', 'aptitude_score', 'communication_score']].mean()
    ax_int.bar(avg_scores.index, avg_scores.values,
           color=['#42a5f5', '#66bb6a', '#ffa726'])  # blue, green, orange
    ax_int.set_title("Average Interview Scores")
    st.pyplot(fig_int)
    # 4️⃣ High Risk Candidates
    st.subheader("⚠️ High Risk Candidates Distribution")
    high_risk = filtered_df[
    (filtered_df['skills_match_percentage'] < 60) |
    (filtered_df['employment_gap_months'] > 12)]
    risk_counts = [len(high_risk), len(filtered_df) - len(high_risk)]
    fig_risk, ax_risk = plt.subplots()
    ax_risk.bar(["High Risk", "Safe"],
                risk_counts,
                color=['#ef5350', '#66bb6a'])  # red & green
    st.pyplot(fig_risk)
    # 5️⃣ Correlation Heatmap
    st.subheader("🔥 Correlation Heatmap")
    import seaborn as sns
    fig2, ax2 = plt.subplots()
    sns.heatmap(filtered_df.corr(), annot=False, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)


# =========================================
# 🎯 PREDICTION
# =========================================
with tab3:

    if model is None:
        st.error("⚠️ model.pkl not found. Train model first.")
    else:
        st.subheader("Predict Placement")

        col1, col2 = st.columns(2)

        with col1:
            tech = st.slider("Technical Score", 0, 100)
            apt = st.slider("Aptitude Score", 0, 100)
            comm = st.slider("Communication Score", 0, 100)

        with col2:
            skills_input = st.slider("Skills Match %", 0, 100)
            experience_input = st.slider("Experience", 0, 10)

        if st.button("Predict"):

            input_data = np.array([[tech, apt, comm, skills_input, experience_input]])
            input_scaled = scaler.transform(input_data)

            pred = model.predict(input_scaled)[0]
            proba = model.predict_proba(input_scaled)[0][1]

            # Result
            if pred == 1:
                st.success("✅ Placed")
            else:
                st.error("❌ Not Placed")

            # Confidence
            st.markdown(f"### 🎯 Confidence: {proba*100:.2f}%")

            # Chart
            fig6, ax6 = plt.subplots()
            ax6.bar(["Placed", "Not Placed"],
                    [1 if pred == 1 else 0, 1 if pred == 0 else 0],
                    color=['#66bb6a', '#ef5350'])
            st.pyplot(fig6)

        # =========================================
        # 📊 MODEL PERFORMANCE
        # =========================================
        st.markdown("### 📊 Model Performance")

        X = filtered_df[['technical_score', 'aptitude_score', 'communication_score',
                         'skills_match_percentage', 'years_of_experience']]
        y = filtered_df['status']

        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)

        # Confusion Matrix
        cm = confusion_matrix(y, y_pred)
        fig_cm, ax_cm = plt.subplots()
        ConfusionMatrixDisplay(cm, display_labels=["Not Placed", "Placed"]).plot(
            cmap="GnBu", ax=ax_cm, colorbar=False)
        st.pyplot(fig_cm)

        # Metrics
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)

        m1, m2, m3 = st.columns(3)
        m1.metric("Precision", f"{precision:.2f}")
        m2.metric("Recall", f"{recall:.2f}")
        m3.metric("F1 Score", f"{f1:.2f}")

        # ROC Curve
        y_prob = model.predict_proba(X_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_prob)
        roc_auc = auc(fpr, tpr)

        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, color='#42a5f5', label=f"AUC = {roc_auc:.2f}")
        ax_roc.plot([0, 1], [0, 1], linestyle='--', color='gray')
        ax_roc.legend()
        st.pyplot(fig_roc)

# =========================================
# 📋 DATA
# =========================================
if st.checkbox("Show Data"):
    st.dataframe(filtered_df.head(100))

    