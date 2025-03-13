import streamlit as st
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# Expanded Career Data with Additional Fields
data = [
    [8, 9, 5, 4, 6, "Data Science", "IITs, IIITs, BITS, Private Universities"],
    [7, 8, 6, 5, 7, "Public Policy", "Ashoka, Jindal, ISPP, DU, TISS"],
    [6, 5, 8, 7, 9, "Environmental Science", "IISc, TISS, DU, JNU"],
    [5, 6, 9, 8, 4, "Development Sector", "TISS, Azim Premji University, IRMA"],
    [9, 8, 4, 3, 5, "Statistics & Mathematics", "ISI, CMI, IITs, DU"],
    [7, 7, 7, 6, 8, "Climate Science Policy", "IISER, IITs, DU, JNU"],
    [6, 8, 5, 7, 9, "Sustainability Consulting", "TERI, Ashoka, JNU, IIMs"],
    [8, 6, 7, 5, 7, "Behavioral Economics", "Ashoka, Jindal, IIMs, DU"],
    [7, 7, 6, 8, 5, "Social Entrepreneurship", "TISS, IRMA, IIMs, Ashoka"],
    [5, 5, 9, 9, 6, "Human Rights & Advocacy", "NLUs, JNU, DU, TISS"],
    [6, 7, 8, 5, 6, "Linguistics", "JNU, DU, EFLU, Ashoka"],
    [5, 6, 7, 7, 5, "Philosophy", "JNU, DU, Ashoka, Jindal"],
    [8, 5, 9, 6, 7, "Design & Visual Arts", "NID, NIFT, MIT-ID, Srishti"],
    [7, 8, 6, 7, 9, "Performing Arts", "NSD, FTII, NID, RADA"],
]

df = pd.DataFrame(data, columns=["Logic", "Analytical", "Creativity", "Social", "Environmental", "Career", "Colleges"])
X = df.iloc[:, :5]
y = df["Career"]

# Encoding career labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Models
tree_clf = DecisionTreeClassifier()
knn_clf = KNeighborsClassifier(n_neighbors=3)
tree_clf.fit(X, y_encoded)
knn_clf.fit(X, y_encoded)

# Streamlit UI
st.set_page_config(page_title="Career Mapping", layout="wide")
st.title("🌟 AI-Powered Career Guidance System")
st.write("#### Discover the best career paths based on your strengths!")

# Input sliders
logic = st.slider("Logical Thinking (1-10)", 1, 10, 5)
analytical = st.slider("Analytical Ability (1-10)", 1, 10, 5)
creativity = st.slider("Creativity (1-10)", 1, 10, 5)
social = st.slider("Social Engagement (1-10)", 1, 10, 5)
environmental = st.slider("Environmental Awareness (1-10)", 1, 10, 5)

# Predict button
if st.button("Find My Career Path 🚀"):
    user_input = np.array([[logic, analytical, creativity, social, environmental]])
    career_tree = encoder.inverse_transform(tree_clf.predict(user_input))[0]
    career_knn = encoder.inverse_transform(knn_clf.predict(user_input))[0]
    
    # Get college details
    college_tree = df[df['Career'] == career_tree]['Colleges'].values[0]
    college_knn = df[df['Career'] == career_knn]['Colleges'].values[0]
    
    st.success(f"🎯 **Top Career Recommendation (Decision Tree):** {career_tree}")
    st.info(f"🏛 **Best Colleges:** {college_tree}")
    
    st.success(f"🎯 **Alternative Career Suggestion (KNN):** {career_knn}")
    st.info(f"🏛 **Best Colleges:** {college_knn}")

st.write("---")
st.write("📌 **This AI-driven tool is designed to help students from rural areas find suitable career paths based on their unique skills and interests.**")

