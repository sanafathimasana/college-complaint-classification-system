
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="College Complaint Classification System",
    page_icon="🎓",
    layout="wide"
)

# ---------------- LOAD MODEL AND DATA ----------------

@st.cache_resource
def load_model():
    return joblib.load("complaint_classifier.pkl")

model = load_model()

DATA_URL = (
    "https://huggingface.co/datasets/sanchit-19/"
    "university-students-complaints/resolve/main/train.csv"
)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL)

df = load_data()

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
    .stApp {
        background-color: #f4f7fb;
    }

    .main-title {
        font-size: 32px;
        font-weight: 700;
        color: #183153;
    }

    .subtitle {
        font-size: 16px;
        color: #64748b;
    }

    .result-box {
        padding: 22px;
        background-color: #e0f2fe;
        border-left: 6px solid #0284c7;
        border-radius: 12px;
        margin-top: 15px;
    }

    .result-label {
        font-size: 15px;
        color: #334155;
    }

    .result-value {
        font-size: 26px;
        font-weight: bold;
        color: #075985;
    }

    div.stButton > button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 22px;
        font-weight: bold;
    }

    div.stButton > button:hover {
        background-color: #1d4ed8;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------

st.sidebar.title("🎓 Complaint Portal")
st.sidebar.write("College Complaint Classification System")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home Dashboard",
        "Predict Department",
        "Department Statistics",
        "About Project"
    ]
)

st.sidebar.divider()
st.sidebar.caption("AI-powered complaint classification")

# ---------------- HOME DASHBOARD ----------------

if page == "Home Dashboard":

    st.markdown(
        '<div class="main-title">College Complaint Classification System</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Classify student complaints and identify the responsible department.</div>',
        unsafe_allow_html=True
    )

    st.write("")

    total_complaints = len(df)
    total_departments = df["Primary_Department"].nunique()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Complaints", total_complaints)

    with col2:
        st.metric("Departments", total_departments)

    with col3:
        st.metric("Linear SVM Test Accuracy", "98.15%")

    st.write("")

    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("Complaints by Department")

        counts = df["Primary_Department"].value_counts()

        fig, ax = plt.subplots(figsize=(9, 5))
        counts.sort_values().plot(
            kind="barh",
            ax=ax
        )

        ax.set_xlabel("Number of Complaints")
        ax.set_ylabel("Department")
        ax.set_title("Department-wise Complaint Distribution")
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("Quick Complaint Prediction")

        quick_complaint = st.text_area(
            "Enter your complaint",
            placeholder="Example: The Wi-Fi is not working in the computer lab.",
            height=150,
            key="home_complaint"
        )

        if st.button("Predict Department", key="home_predict"):

            if quick_complaint.strip():

                prediction = model.predict([quick_complaint])[0]

                st.success("Complaint classified successfully!")

                st.markdown(
                    f"""
                    <div class="result-box">
                        <div class="result-label">Predicted Department</div>
                        <div class="result-value">{prediction}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:
                st.warning("Please enter a complaint first.")

    st.subheader("Recent Dataset Records")

    display_columns = [
        "Complaint_Description",
        "Primary_Department"
    ]

    st.dataframe(
        df[display_columns].head(5),
        use_container_width=True,
        hide_index=True
    )

# ---------------- PREDICT DEPARTMENT ----------------

elif page == "Predict Department":

    st.title("📝 Predict Complaint Department")

    st.write(
        "Enter the student's complaint below. "
        "The trained machine-learning model will predict the responsible department."
    )

    complaint = st.text_area(
        "Complaint Description",
        placeholder=(
            "Example: The college bus is arriving late every day "
            "and students are unable to reach class on time."
        ),
        height=180
    )

    if st.button("Classify Complaint"):

        if complaint.strip():

            with st.spinner("Analyzing complaint..."):

                prediction = model.predict([complaint])[0]

            st.success("Prediction completed!")

            st.markdown(
                f"""
                <div class="result-box">
                    <div class="result-label">Predicted Department</div>
                    <div class="result-value">{prediction}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.info(
                f"The complaint should be reviewed by the "
                f"**{prediction}** department."
            )

        else:
            st.warning("Please enter a complaint before clicking Classify Complaint.")

# ---------------- DEPARTMENT STATISTICS ----------------

elif page == "Department Statistics":

    st.title("📊 Department Statistics")

    st.write("Explore the complaint distribution in the dataset.")

    counts = df["Primary_Department"].value_counts()

    st.bar_chart(counts)

    stats = counts.rename_axis("Department").reset_index(
        name="Number of Complaints"
    )

    st.subheader("Detailed Statistics")

    st.dataframe(
        stats,
        use_container_width=True,
        hide_index=True
    )

    csv = stats.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Department Statistics",
        data=csv,
        file_name="department_statistics.csv",
        mime="text/csv"
    )

# ---------------- ABOUT PROJECT ----------------

elif page == "About Project":

    st.title("ℹ️ About the Project")

    st.write("""
    ### College Complaint Classification System

    This project uses Natural Language Processing (NLP) and machine
    learning to classify student complaints into responsible departments.

    **Objectives**
    - Automatically classify student complaints.
    - Identify the department responsible for reviewing each complaint.
    - Reduce manual complaint sorting.
    - Provide department-wise complaint statistics.

    **Technology Used**
    - Python
    - Pandas
    - Scikit-learn
    - TF-IDF text vectorization
    - Linear Support Vector Machine (SVM)
    - Streamlit
    """)

    st.warning(
        "Predictions are suggestions for routing complaints. "
        "The responsible department should review each complaint."
    )

st.sidebar.divider()
st.sidebar.caption("College Complaint Classification System | Academic Project")
