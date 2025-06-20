import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Load CSV file
df = pd.read_csv("reviews.csv")
texts = df["text"].tolist()
labels = df["label"].tolist()

# Vectorize and train
vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
X = vectorizer.fit_transform(texts)
y = labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = MultinomialNB()
model.fit(X_train, y_train)

# Streamlit UI
st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Enter a movie review below:")

user_input = st.text_area("Type your review here:")

if st.button("Analyze"):
    if user_input:
        vector = vectorizer.transform([user_input])
        prediction = model.predict(vector)[0]
        if prediction == "pos":
            st.success("✅ Positive review! 🎉")
        else:
            st.error("❌ Negative review. 😞")
    else:
        st.warning("Please type something to analyze.")
