import streamlit as st
import cv2
import numpy as np
import tempfile
from roboflow import Roboflow

#  Initialize Roboflow Model
rf = Roboflow(api_key="fp0VkETISZJl5bIRORHx")
project = rf.project("project1-bb4os")  
model = project.version("1").model

#  Streamlit UI
st.title("🚗 Real-time Vehicle Detection")
st.sidebar.header("📷 Webcam Settings")

#  Start/Stop Webcam Button
start_detection = st.sidebar.button("Start Detection")

FRAME_WINDOW = st.empty()  # Streamlit placeholder for displaying frames

if start_detection:
    cap = cv2.VideoCapture(0)  # Ensure this is set to the correct webcam index

    #  Check if the camera opens properly
    if not cap.isOpened():
        st.error("⚠️ Failed to open webcam. Please check your camera permissions or try a different camera index.")
    else:
        st.success("✅ Webcam started successfully!")

        while True:
            ret, frame = cap.read()

            if not ret:
                st.error("⚠️ Could not read the frame from webcam. Trying again...")
                continue  # Try again instead of breaking the loop

            #  Save frame as a temporary file (Fix for Roboflow API)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_filename = temp_file.name
                cv2.imwrite(temp_filename, frame)

            #  Perform Prediction using the file path
            try:
                response = model.predict(temp_filename, confidence=40, overlap=30).json()
            except Exception as e:
                st.error(f"⚠️ Error during prediction: {e}")
                continue

            #  Draw Bounding Boxes on the frame
            for prediction in response.get('predictions', []):
                x, y, width, height = int(prediction['x']), int(prediction['y']), int(prediction['width']), int(prediction['height'])
                label = prediction['class']

                cv2.rectangle(frame, (x - width // 2, y - height // 2), 
                              (x + width // 2, y + height // 2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            #  Convert frame to RGB and display in Streamlit
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame, channels="RGB")

        cap.release()
        cv2.destroyAllWindows()
