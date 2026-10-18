from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import streamlit as st
import pandas as pd
import numpy as np

columns = ["buying","maint","doors","persons","lug_boot","safety","class"]
df = pd.read_csv("car.data", names=columns)

new_df = df.drop("class", axis=1)

st.set_page_config(page_title="Car Evaluation Model", layout="wide")
st.title("Car Evaluation Model", anchor=False)

feature_columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]

buying = st.selectbox("choose a buying method", ["vhigh", "high", "med", "low"], index=None, placeholder="select a method....")
maint = st.selectbox("choose a maint", ["vhigh", "high", "med", "low"], index=None, placeholder="select a method....")
doors = st.selectbox("choose the number of doors", ["2", "3", "4", "5more"], index=None, placeholder="select doors....")
persons = st.selectbox("choose the number of persons", ["2", "4", "more"], index=None, placeholder="select persons....")
lug_boot = st.selectbox("choose a lug_boot", ["small", "med", "big"], index=None, placeholder="select a method....")
safety = st.selectbox("choose a safety", ["low", "med", "high"], index=None, placeholder="select a safety method....")

X = pd.get_dummies(new_df, columns=feature_columns, dtype=int)

lab = LabelEncoder()
y = lab.fit_transform(df["class"])

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

model = make_pipeline(LogisticRegression(max_iter=2000))
model.fit(X_train, y_train)

if st.button("Predict"):
    if any(value is None for value in [buying, maint, doors, persons, lug_boot, safety]):
        st.warning("Please fill in all fields before predicting.")
        st.stop()

    user_input = pd.DataFrame([{
        "buying": buying,
        "maint": maint,
        "doors": doors,
        "persons": persons,
        "lug_boot": lug_boot,
        "safety": safety,
    }])

    user_input = pd.get_dummies(user_input, columns=feature_columns, dtype=int)
    user_input = user_input.reindex(columns=X.columns, fill_value=0)

    prediction = model.predict(user_input)[0]
    st.write("The prediction is:", lab.inverse_transform([prediction])[0])

df