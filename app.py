import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical
import streamlit as st
from PIL import Image
import cv2

# Load digits dataset
digits = load_digits()
df = digits.images
target = digits.target

# Streamlit App
st.title("Handwritten Digit Classifier")

# Display dataset samples
if st.checkbox("Show Dataset Samples"):
    st.write("Sample Digits from Dataset")
    plt.figure(figsize=(10,10))
    s = 1
    for i in range(10):
        plt.subplot(5,2,s)
        plt.imshow(df[i], cmap="gray", interpolation="bicubic")
        plt.xticks([])
        plt.yticks([])
        s += 1
    st.pyplot(plt)

# Data Preprocessing
X = df
y = target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)
w, h = df[0].shape
X_train = X_train.reshape(len(X_train), w * h)
X_test = X_test.reshape(len(X_test), w * h)
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# Build the model
model = Sequential()
model.add(Dense(64, input_shape=(w * h,), activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(10, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
history = model.fit(X_train, y_train_cat, batch_size=32, epochs=20, validation_split=0.1)

# Function for predictions and visualization
def Prediction_cat(model, image):
    pred = model.predict(image)
    color = ["red"] * 10
    num = pred.argmax()
    color[num] = "blue"
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Display the image
    ax1.imshow(image.reshape(w, h), cmap="gray", interpolation="bicubic")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title("Input Image")

    # Display the prediction as a bar chart
    ax2.bar(x=[0,1,2,3,4,5,6,7,8,9], height=pred.flatten(), color=color)
    ax2.set_xlabel("Digits")
    ax2.set_ylabel("Probability")
    ax2.set_title("Prediction")
    ax2.set_xticks([0,1,2,3,4,5,6,7,8,9])

    st.pyplot(fig)

# Input image upload
uploaded_file = st.file_uploader("Upload an image of a handwritten digit", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read the image
    img = Image.open(uploaded_file)
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((w, h))  # Resize to match dataset size

    # Display uploaded image
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img_arr = np.array(img)
    img_arr = img_arr.reshape(1, w * h)

    # Predict the digit
    Prediction_cat(model, img_arr)

# Model evaluation and accuracy plots
if st.checkbox("Show Training Progress"):
    st.write("Model Performance")
    result = model.evaluate(X_test.reshape(len(X_test), w * h), y_test_cat)
    for i in range(len(model.metrics_names)):
        st.write(f"{model.metrics_names[i]}: {result[i]}")

    st.line_chart(history.history["acc"], width=0, height=0, use_container_width=True)
    st.line_chart(history.history["val_acc"], width=0, height=0, use_container_width=True)
