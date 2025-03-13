import streamlit as st
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# Title and Description
st.title("AI-Driven Career Mapping System")
st.write("### Discover Your Ideal Career Path Based on Your Personality, Interests, and Aptitude")

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
    [7, 8, 9, 6, 5, "Actuarial Science"],
    [6, 5, 7, 9, 8, "Nanotechnology"],
    [8, 6, 5, 7, 9, "Forensic Science"],
    [7, 9, 6, 8, 5, "Education Technology"],
    [9, 7, 8, 6, 5, "Wine Technology"],
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

# Questions for User Input (Scenario-Based & MBTI Style)
st.write("## Answer the Following Questions to Get Your Career Recommendation")
q1 = st.radio("When given a complex problem, you prefer:", ["Breaking it into small logical steps", "Discussing it with a team", "Thinking creatively about solutions", "Analyzing data and trends"])
q2 = st.radio("You are happiest when:", ["Solving puzzles and logical problems", "Expressing your creativity", "Helping people and making an impact", "Learning new theories and researching"])
q3 = st.radio("What kind of environment do you thrive in?", ["Structured and analytical", "Creative and open-ended", "Social and interactive", "Independent and research-driven"])
q4 = st.radio("Which statement describes you best?", ["I enjoy working with numbers and data", "I like to think outside the box", "I enjoy social impact work", "I am fascinated by technology and innovation"])
q5 = st.radio("Which task sounds most interesting to you?", ["Designing algorithms", "Writing a book or making art", "Developing policies for social good", "Conducting scientific research"])

# Convert Answers to Numerical Data
responses = {
    "Breaking it into small logical steps": 9,
    "Discussing it with a team": 6,
    "Thinking creatively about solutions": 7,
    "Analyzing data and trends": 8,
    "Solving puzzles and logical problems": 9,
    "Expressing your creativity": 7,
    "Helping people and making an impact": 6,
    "Learning new theories and researching": 8,
    "Structured and analytical": 9,
    "Creative and open-ended": 7,
    "Social and interactive": 6,
    "Independent and research-driven": 8,
    "I enjoy working with numbers and data": 9,
    "I like to think outside the box": 7,
    "I enjoy social impact work": 6,
    "I am fascinated by technology and innovation": 8,
    "Designing algorithms": 9,
    "Writing a book or making art": 7,
    "Developing policies for social good": 6,
    "Conducting scientific research": 8,
}

user_data = np.array([[responses[q1], responses[q2], responses[q3], responses[q4], responses[q5]]])

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
        body { background-color: #f4f4f4; }
    </style>
    """, unsafe_allow_html=True)
