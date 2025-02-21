import streamlit as st
import cv2
import torch
from PIL import Image
import numpy as np
import tempfile
import pathlib

# Fix for Windows path issue
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Set page title and icon
st.set_page_config(page_title="YOLOv5 Live Object Detection", page_icon="🎥")

# Load YOLOv5 model
@st.cache_resource
def load_model(weights_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Load trained YOLOv5 model"""
    try:
        # Load model
        model = torch.hub.load('./yolov5', 'custom', path=weights_path, source='local')
        model.to(device)
        model.eval()
        st.success(f"Model loaded successfully on {device}")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to perform object detection and visualization
def detect_and_visualize(model, frame, conf_threshold=0.70):
    """
    Detect objects and visualize results with bounding boxes.
    """

    global mode
    
    try:
        # Convert frame to RGB (YOLOv5 expects RGB images)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run detection
        model.conf = conf_threshold
        results = model(frame_rgb)

        # Draw bounding boxes and labels
        for det in results.pred[0]:
            x1, y1, x2, y2, conf, cls = det.cpu().numpy()
            class_name = results.names[int(cls)]

            if mode == "Image":
                # Set color based on class
                color = (255, 0, 0) if class_name.lower() == 'cheating' else (0, 255, 0)
            else:
                color = (0, 0, 255) if class_name.lower() == 'cheating' else (0, 255, 0)

            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # Add label with confidence
            label = f'{class_name} {conf:.2f}'
            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame

    except Exception as e:
        st.error(f"Error during detection: {str(e)}")
        return frame

mode = st.sidebar.selectbox("Select Mode", ["Webcam", "Image", "Video"])

# Main function for the Streamlit app
def main():
    """
    Main function to run the Streamlit app.
    """

    global mode

    st.title("Exam cheating Detection App")
    st.sidebar.title("Settings")

    # Mode selection
    # Confidence threshold slider
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.70, 0.01)

    # Start/stop webcam button
    run = st.sidebar.checkbox("Start Webcam")

    # Placeholder for displaying the webcam feed
    FRAME_WINDOW = st.image([])

    # Load the YOLOv5 model
    weights_path = './best.pt'  # Update this path to your model weights
    model = load_model(weights_path)

    if model is None:
        st.error("Failed to load model. Please check the model path.")
        return

    if mode == "Webcam":
        # Access the webcam
        cap = cv2.VideoCapture(0)

        while run:
            # Read a frame from the webcam
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture video. Please check your webcam.")
                break

            # Perform object detection on the frame
            frame = detect_and_visualize(model, frame, confidence_threshold)

            # Convert the frame from BGR to RGB for display in Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display the frame in the Streamlit app
            FRAME_WINDOW.image(frame_rgb, channels="RGB", use_container_width=True)

        # Release the webcam when the app stops
        cap.release()

    elif mode == "Image":
        # Image mode
        st.header("Image Upload")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Read the uploaded image
            image = Image.open(uploaded_file)
            frame = np.array(image)  # Convert PIL image to numpy array

            # Perform object detection and visualization
            frame = detect_and_visualize(model, frame, confidence_threshold)

            # Display the image with detections
            st.image(frame, channels="RGB", use_container_width=True)
            
    elif mode == "Video":
        # Video mode
        st.header("Video Upload")
        uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

        if uploaded_file is not None:
            # Save the uploaded video to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
                tfile.write(uploaded_file.read())
                video_path = tfile.name

            # Open the video file
            cap = cv2.VideoCapture(video_path)

            # Placeholder for displaying the video feed
            FRAME_WINDOW = st.image([])

            while cap.isOpened():
                # Read a frame from the video
                ret, frame = cap.read()
                if not ret:
                    break

                # Perform object detection and visualization
                frame = detect_and_visualize(model, frame, confidence_threshold)

                # Convert the frame from BGR to RGB for display in Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Display the frame in the Streamlit app
                FRAME_WINDOW.image(frame_rgb, channels="RGB", use_container_width=True)

            # Release the video capture object
            cap.release()

# Run the app
if __name__ == "__main__":
    main()