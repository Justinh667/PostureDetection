from flask import Flask, render_template_string, Response
from ultralytics import YOLO
import cv2
import jetson.inference
import jetson.utils
import numpy as np

# Flask app + YOLO model
app = Flask(__name__)
model = YOLO('/home/nvidia10/Posture_Detection/runs/classify/train/weights/best.pt')

# Load PoseNet with hand keypoints
posenet = jetson.inference.poseNet("resnet18-hand", threshold=0.15)

# Open webcam
cap = cv2.VideoCapture(0)

# Updated HTML title to "Posture Detection"
HTML = '''
<!doctype html>
<html>
<head><title>Posture Detection</title></head>
<body style="text-align:center; background:#f4f4f4; font-family:sans-serif;">
    <h1>Posture Detection</h1>
    <img src="/video" width="720">
</body>
</html>
'''

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # YOLO classification
        result = model.predict(frame, verbose=False)[0]
        top1_id = result.probs.top1
        confidence = float(result.probs.top1conf)
        label = result.names[top1_id].lower()
        percent = int(confidence * 100)

        if confidence < 0.65:
            text = "N/A – No Person Detected"
            color = (128, 128, 128)
        elif "confident" in label:
            text = f"CONFIDENT POSTURE ({percent}%)"
            color = (0, 255, 0)
        elif "lousy" in label:
            text = f"LOUSY POSTURE ({percent}%)"
            color = (0, 0, 255)
        else:
            text = "N/A – Unknown"
            color = (100, 100, 100)

        # Convert to CUDA image for PoseNet
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cuda_img = jetson.utils.cudaFromNumpy(img_rgb)

        # Run PoseNet (hand or body depending on model)
        poses = posenet.Process(cuda_img)

        # Draw all available keypoints
        for pose in poses:
            for keypoint in pose.Keypoints:
                if keypoint.ID >= 0:
                    jetson.utils.cudaDrawCircle(cuda_img, (int(keypoint.x), int(keypoint.y)), 4, (255, 255, 0, 255))

        # Convert back to numpy for web stream
        np_img = jetson.utils.cudaToNumpy(cuda_img)
        np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

        # Overlay text
        cv2.putText(np_img, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)

        # Stream over HTTP
        ret, buffer = cv2.imencode('.jpg', np_img)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Launch the server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
