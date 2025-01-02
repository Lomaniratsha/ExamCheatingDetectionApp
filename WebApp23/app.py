from flask import Flask, render_template, request, Response, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load YOLO model
model = YOLO('best.pt')

# Initialize camera variable
camera = None

def process_frame(frame):
    # Run YOLOv8 inference on the frame
    results = model(frame)
    
    # Process the results
    annotated_frame = results[0].plot()
    return annotated_frame

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save the uploaded file
    filename = f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Read and process the image
    frame = cv2.imread(filepath)
    processed_frame = process_frame(frame)
    
    # Save the processed image
    output_filename = f"processed_{filename}"
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    cv2.imwrite(output_path, processed_frame)

    return jsonify({'processed_image': f"uploads/{output_filename}"})

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save the uploaded file
    filename = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Process the video
    cap = cv2.VideoCapture(filepath)
    output_filename = f"processed_{filename}"
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create VideoWriter object
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        processed_frame = process_frame(frame)
        out.write(processed_frame)

    cap.release()
    out.release()

    return jsonify({'processed_video': f"uploads/{output_filename}"})

def generate_frames():
    global camera
    
    while True:
        if camera is None:
            break
            
        success, frame = camera.read()
        if not success:
            break
        
        processed_frame = process_frame(frame)
        
        # Convert to jpg format
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera')
def start_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    return jsonify({'status': 'success'})

@app.route('/stop_camera')
def stop_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True) 