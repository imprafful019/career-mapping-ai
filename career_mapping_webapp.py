import streamlit as st
import numpy as np
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# Expanded Career Fields Data (Logical, Analytical, Creative, Social, Environmental Interest) with career paths
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
    [5, 5, 9, 9, 6, "Human Rights & Advocacy"]
])

X = data[:, :-1].astype(float)  # Features
Y = data[:, -1]  # Career labels
encoder = LabelEncoder()
Y_encoded = encoder.fit_transform(Y)

# Train Decision Tree Model
career_tree = DecisionTreeClassifier()
career_tree.fit(X, Y_encoded)

# Train KNN Model
career_knn = KNeighborsClassifier(n_neighbors=3)
career_knn.fit(X, Y_encoded)

# Streamlit Web App
st.title("AI-Driven Career Mapping System")
st.write("Answer the following questions to find your ideal career path!")

# User Inputs
logical = st.slider("Logical Thinking (1-10)", 1, 10, 5)
analytical = st.slider("Analytical Skills (1-10)", 1, 10, 5)
creative = st.slider("Creativity (1-10)", 1, 10, 5)
social = st.slider("Social Skills (1-10)", 1, 10, 5)
environmental = st.slider("Interest in Environmental Issues (1-10)", 1, 10, 5)

if st.button("Get Career Recommendation"):
    features = np.array([[logical, analytical, creative, social, environmental]])
    decision_tree_result = encoder.inverse_transform(career_tree.predict(features))
    knn_result = encoder.inverse_transform(career_knn.predict(features))
    
    st.success(f"Decision Tree Suggests: {decision_tree_result[0]}")
    st.success(f"KNN Suggests: {knn_result[0]}")
