import streamlit as st
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# Title and Description
st.title("AI-Driven Career Mapping System")
st.write("### Discover Your Ideal Career Path Based on Your Skills, Interests, and Aptitude")

# Expanded Career Fields Data
data = np.array([
    [8, 9, 5, 4, 6, "Data Science"],
    [7, 8, 6, 5, 7, "Public Policy"],
    [6, 5, 8, 7, 9, "Environmental Science"],
    [5, 6, 9, 8, 4, "Development Sector"],
    [9, 8, 4, 3, 5, "Statistics & Mathematics"],
    [7, 7, 7, 6, 8, "Climate Science Policy"],
    [6, 8, 5, 7, 9, "Sustainability Consulting"],
    [8, 6, 7, 5, 7, "Behavioral Economics"],
    [7, 7, 6, 8, 5, "Social Entrepreneurship"],
    [5, 5, 9, 9, 6, "Human Rights & Advocacy"],
    [6, 7, 8, 5, 6, "Linguistics"],
    [5, 6, 7, 7, 5, "Philosophy"],
    [8, 5, 9, 6, 7, "Design & Visual Arts"],
    [6, 9, 8, 5, 7, "Performing Arts"],
    [7, 8, 6, 7, 8, "Anthropology"],
    [5, 7, 9, 8, 6, "Archaeology"],
])

# Label Encoding
le = LabelEncoder()
data[:, -1] = le.fit_transform(data[:, -1])
data = data.astype(float)

# Feature Extraction & Model Training
X = data[:, :-1]
y = data[:, -1]

dt_model = DecisionTreeClassifier()
dt_model.fit(X, y)

knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X, y)

# Questions for User Input
st.write("## Answer the Following Questions to Get Your Career Recommendation")
skill_analytical = st.radio("How strong are your analytical skills?", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
interest_creative = st.radio("How interested are you in creative fields?", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
passion_social = st.radio("Do you enjoy working in social impact sectors?", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
environment_awareness = st.radio("How concerned are you about environmental issues?", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
logical_thinking = st.radio("How strong are your logical reasoning skills?", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

user_data = np.array([[skill_analytical, interest_creative, passion_social, environment_awareness, logical_thinking]])

# Career Prediction
if st.button("Find My Career Path"):
    dt_prediction = dt_model.predict(user_data)
    knn_prediction = knn_model.predict(user_data)
    
    career_dt = le.inverse_transform(dt_prediction.astype(int))[0]
    career_knn = le.inverse_transform(knn_prediction.astype(int))[0]
    
    st.write(f"### Career Recommendation: {career_dt} or {career_knn}")
    st.write("Your responses indicate that these fields align well with your strengths and interests.")

# Improved UI Styling
st.markdown("""
    <style>
        .stButton>button { border-radius: 10px; background-color: #4CAF50; color: white; padding: 10px 24px; font-size: 16px; }
        .stRadio { font-size: 18px; }
    </style>
    """, unsafe_allow_html=True)

