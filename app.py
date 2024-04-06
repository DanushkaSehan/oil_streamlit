import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('model.h5')

# Define the class names
class_names = ['non-oily', 'oily']

# Function to detect oiliness on face
def detect_oiliness(frame):
    # Preprocess the frame
    frame = cv2.resize(frame, (224, 224))  # Assuming the model requires input size of 224x224
    frame = np.expand_dims(frame, axis=0)   # Add batch dimension
    frame = frame / 255.0                    # Normalize pixel values
    
    # Perform prediction
    predictions = model.predict(frame)
    predicted_class = np.argmax(predictions)
    predicted_label = class_names[predicted_class]
    confidence_level = predictions[0][predicted_class] * 100  # Confidence as percentage

    # Determine OILINESS level
    oiliness_level = None
    if predicted_label == 'oily':
        if confidence_level > 80:
            oiliness_level = "HIGH-OILYNESS"
        elif confidence_level <= 80 and confidence_level > 60:
            oiliness_level = "MID-OILYNESS"
        else:
            oiliness_level = "LOW-OILYNESS"
    elif predicted_label == 'non-oily':
        oiliness_level = "LOW-OILYNESS"
    
    return oiliness_level

def main():
    st.title('Oiliness Detection App')

    # Open webcam
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        return

    # Add a button to capture image
    if st.button("Capture Image"):
        ret, frame = cap.read()
        if ret:
            # Detect oiliness on the captured frame
            oiliness_level = detect_oiliness(frame)

            # Display oiliness level
            st.write("Oiliness Level:", oiliness_level)

            # Display the captured image
            st.image(frame, channels='BGR')

    # Release webcam
    cap.release()

if __name__ == '__main__':
    main()
