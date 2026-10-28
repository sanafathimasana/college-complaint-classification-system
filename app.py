
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="College Complaint Classification System",
    page_icon="🎓",
    layout="wide"
)

# Load trained ML model
@st.cache_resource
def load_model():
    return joblib.load("complaint_classifier.pkl")

model = load_model()

# Load dataset
DATA_URL = "https://huggingface.co/datasets/sanchit-19/university-students-complaints/resolve/main/train.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL)

df = load_data()

# Department keywords
department_keywords = {
    "Canteen": [
        "canteen", "food", "meal", "hygiene",
        "dining area", "food quality", "unclean plates",
        "stale food", "drinking water"
    ],
    "Transport & Parking": [
        "college bus", "bus delay", "bus late",
        "transport", "parking", "bus schedule",
        "vehicle", "bus route"
    ],
    "Maintenance": [
        "fan", "classroom lights", "ceiling",
        "broken furniture", "repair", "leakage",
        "damaged desk", "electrical problem"
    ],
    "IT": [
        "wifi", "wi-fi", "internet", "computer lab",
        "software", "network", "login problem",
        "website not working", "server"
    ],
    "Finance": [
        "fee payment", "semester fee", "refund",
        "payment", "scholarship", "tuition fee",
        "fee receipt", "billing"
    ],
    "Library": [
        "library", "textbook", "books",
        "library hours", "book availability",
        "return books", "library card"
    ],
    "Student Welfare": [
        "counselling", "counseling", "student welfare",
        "mental stress", "student support",
        "student safety", "hostel welfare"
    ],
    "Proctorial": [
        "discipline", "misconduct", "ragging",
        "student behaviour", "student behavior",
        "harassment", "fight between students"
    ],
    "Registrar": [
        "student records", "date of birth",
        "transcript", "certificate", "official records",
        "name correction", "registration number"
    ],
    "Department Head": [
        "head of department", "hod",
        "department timetable", "class schedule",
        "faculty allocation", "department classes",
        "syllabus", "course schedule"
    ]
}

# Prediction function
def predict_department(complaint):
    complaint_lower = complaint.lower().strip()

    # Find matching departments and their keyword counts
    scores = {}

    for department, keywords in department_keywords.items():
        scores[department] = sum(
            1 for keyword in keywords
            if keyword in complaint_lower
        )

    # Use keyword result if a department has matching keywords
    best_department = max(scores, key=scores.get)
    best_score = scores[best_department]

    if best_score > 0:
        return best_department

    # Otherwise use the trained machine learning model
    return model.predict([complaint])[0]


# Sidebar navigation
st.sidebar.title("🎓 College Complaint System")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home Dashboard",
        "Predict Department",
        "Department Statistics",
        "About Project"
    ]
)

# Home Dashboard
if page == "Home Dashboard":
    st.title("🎓 College Complaint Classification System")

    st.write(
        "An intelligent system that classifies college complaints "
        "into the appropriate departments using Machine Learning."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Complaints", len(df))

    with col2:
        st.metric("Departments", df["Primary_Department"].nunique())

    with col3:
        st.metric("Classification Method", "ML + Keywords")

    st.subheader("Complaints by Department")

    department_counts = df["Primary_Department"].value_counts()

    fig, ax = plt.subplots(figsize=(10, 5))
    department_counts.plot(kind="bar", ax=ax)

    ax.set_xlabel("Department")
    ax.set_ylabel("Number of Complaints")
    ax.set_title("Department-wise Complaint Distribution")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.info("Select 'Predict Department' from the sidebar to test a complaint.")

# Predict Department
elif page == "Predict Department":
    st.title("📝 Complaint Department Prediction")

    st.write(
        "Enter your college complaint below. "
        "The system will predict the department responsible for handling it."
    )

    complaint = st.text_area(
        "Enter Your Complaint",
        placeholder="Example: The food served in the college canteen is often cold...",
        height=150
    )

    if st.button("Predict Department"):
        if complaint.strip():
            prediction = predict_department(complaint)

            st.success(f"Predicted Department: {prediction}")

            st.write("**Your Complaint:**")
            st.write(complaint)

        else:
            st.warning("Please enter a complaint before predicting.")

# Department Statistics
elif page == "Department Statistics":
    st.title("📊 Department Statistics")

    department_counts = df["Primary_Department"].value_counts()

    st.subheader("Complaint Count by Department")
    st.dataframe(
        department_counts.rename("Number of Complaints").reset_index()
        .rename(columns={"index": "Department", "Primary_Department": "Department"})
    )

    st.subheader("Department Distribution")

    fig, ax = plt.subplots(figsize=(8, 8))
    department_counts.plot(
        kind="pie",
        autopct="%1.1f%%",
        ax=ax
    )

    ax.set_ylabel("")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# About Project
elif page == "About Project":
    st.title("ℹ️ About the Project")

    st.write("""
    ### College Complaint Classification System

    This project helps students identify the appropriate department
    for handling college-related complaints.

    **Technologies Used:**
    - Python
    - Streamlit
    - Pandas
    - Scikit-learn
    - Machine Learning
    - Keyword-based classification

    **Departments Covered:**
    - Maintenance
    - IT
    - Department Head
    - Transport & Parking
    - Finance
    - Student Welfare
    - Library
    - Proctorial
    - Registrar
    - Canteen

    The system uses keyword matching for recognizable complaint topics
    and falls back to the trained machine learning model when no
    configured keyword matches.
    """)

