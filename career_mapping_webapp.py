import streamlit as st
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# Title and Description
st.set_page_config(page_title="AI-Driven Career Mapping", layout="wide")
st.title("🌟 AI-Driven Career Mapping System")
st.write("#### Answer the following questions to find your ideal career path!")

# Career Fields Data with Key Skills Mapping
data = np.array([
    [3, 3, 2, 3, 2, 5, 2, 1, "Data Science"],
    [2, 3, 2, 4, 2, 2, 4, 2, "Public Policy"],
    [2, 2, 4, 4, 2, 5, 2, 1, "Environmental Science"],
    [1, 2, 4, 4, 1, 2, 4, 2, "Development Sector"],
    [3, 3, 3, 3, 2, 5, 2, 2, "Statistics & Mathematics"],
    [2, 2, 2, 2, 2, 5, 2, 1, "Climate Science Policy"],
    [2, 3, 2, 4, 2, 2, 4, 2, "Sustainability Consulting"],
    [3, 3, 2, 2, 2, 5, 2, 2, "Behavioral Economics"],
    [2, 2, 4, 4, 2, 2, 4, 2, "Social Entrepreneurship"],
    [1, 2, 2, 4, 2, 5, 2, 2, "Human Rights & Advocacy"],
    [2, 2, 2, 2, 2, 5, 2, 1, "Linguistics"],
    [1, 2, 2, 2, 2, 5, 2, 1, "Philosophy"],
    [1, 2, 3, 2, 1, 1, 2, 1, "Design & Visual Arts"],
    [1, 2, 3, 4, 1, 1, 2, 1, "Performing Arts"],
    [2, 2, 2, 4, 2, 5, 2, 1, "Anthropology"],
    [2, 2, 2, 2, 2, 5, 2, 1, "Archaeology"],
    [3, 3, 3, 3, 2, 5, 2, 2, "Actuarial Science"],
    [3, 3, 2, 1, 2, 5, 2, 2, "Nanotechnology"],
    [3, 3, 2, 2, 2, 5, 2, 2, "Forensic Science"],
    [2, 2, 2, 4, 2, 2, 4, 2, "Education Technology"],
    [1, 2, 3, 4, 1, 1, 2, 1, "Wine Technology"]
])

# Label Encoding Careers
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
st.write("### Answer the Following Questions")
q1 = st.radio("When working on a project, you prefer:", ["Planning every step in advance", "Experimenting and adjusting as you go", "Balancing structured and flexible approaches"])
q2 = st.radio("When solving problems, you rely on:", ["Logic and analysis", "Creativity and intuition", "A mix of both"])
q3 = st.radio("You are more drawn to:", ["Numbers and data", "People and emotions", "A blend of both"])
q4 = st.radio("Your ideal work environment is:", ["Structured and organized", "Flexible and spontaneous", "A mix of both"])
q5 = st.radio("Which activity do you enjoy the most?", ["Solving complex puzzles", "Creating art or music", "Helping people", "Exploring new ideas and research", "A mix of various activities"])
q6 = st.radio("How do you approach complex challenges?", ["Solving complex puzzles", "Using creativity and intuition", "A mix of both"])
q7 = st.radio("Do you enjoy taking leadership roles?", ["Leading and managing teams", "Collaborating as part of a team", "Working independently"])
q8 = st.radio("How well do you adapt to new situations?", ["Adapting to new situations easily", "Finding structured approaches more comfortable", "A mix of both"])

# Convert Answers to Numerical Data
responses = {
    "Planning every step in advance": 3,
    "Experimenting and adjusting as you go": 1,
    "Balancing structured and flexible approaches": 2,
    "Logic and analysis": 3,
    "Creativity and intuition": 1,
    "A mix of both": 2,
    "Numbers and data": 3,
    "People and emotions": 1,
    "A blend of both": 2,
    "Structured and organized": 3,
    "Flexible and spontaneous": 1,
    "Solving complex puzzles": 5,
    "Creating art or music": 3,
    "Helping people": 4,
    "Exploring new ideas and research": 2,
    "A mix of various activities": 1,
    "Leading and managing teams": 3,
    "Collaborating as part of a team": 2,
    "Working independently": 1,
    "Adapting to new situations easily": 3,
    "Finding structured approaches more comfortable": 1
}

user_data = np.array([[responses[q1], responses[q2], responses[q3], responses[q4], responses[q5], responses[q6], responses[q7], responses[q8]]])

# Career Prediction
if st.button("Find My Career Path"):
    dt_prediction = dt_model.predict(user_data)
    knn_prediction = knn_model.predict(user_data)
    
    career_dt = le.inverse_transform(dt_prediction.astype(int))[0]
    career_knn = le.inverse_transform(knn_prediction.astype(int))[0]
    
    st.success(f"🎯 Best Career Match: {career_dt}")
    st.info(f"🔍 Alternative Career Option: {career_knn}")

st.write("📌 **This AI-driven tool helps students find the best career paths based on their skills and interests.**")
