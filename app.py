import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image

# Load the dataset
df = pd.read_csv("IRIS.csv")

# Encode labels
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])

# Split the data
X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Reverse map label to species
species_map = dict(zip(le.transform(le.classes_), le.classes_))

# Get flower image based on prediction
# Convert class name to filename format
def get_flower_image(species_name):
    filename = species_name.replace(" ", "-") + ".jpg"
    img_path = f"images/{filename}"
    try:
        return Image.open(img_path)
    except FileNotFoundError:
        return None


# Streamlit UI
st.title("🌸 Iris Flower Predictor")

# Input sliders
sepal_length = st.slider("Sepal Length (cm)", float(df['sepal_length'].min()), float(df['sepal_length'].max()))
sepal_width = st.slider("Sepal Width (cm)", float(df['sepal_width'].min()), float(df['sepal_width'].max()))
petal_length = st.slider("Petal Length (cm)", float(df['petal_length'].min()), float(df['petal_length'].max()))
petal_width = st.slider("Petal Width (cm)", float(df['petal_width'].min()), float(df['petal_width'].max()))

# Predict
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_data)[0]
predicted_species = species_map[prediction]

# Output
st.subheader("🔍 Predicted Species:")
st.success(predicted_species)

image = get_flower_image(predicted_species)
if image:
    st.image(image, caption=predicted_species, use_container_width=True)
else:
    st.warning("No image available for this species.")
