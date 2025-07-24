# livestream_posture.py

from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO

app = Flask(__name__)
camera = cv2.VideoCapture("v4l2src device=/dev/video1 ! video/x-raw, width=640, height=480 ! videoconvert ! appsink", cv2.CAP_GSTREAMER)
model = YOLO('/home/nvidia10/Posture_Detection/runs/classify/train/weights/best.pt')

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            print("❌ Failed to read frame from camera")
            break
        else:
            print("✅ Frame read successfully")
        ...

        # Resize frame to model input size (if needed)
        resized = cv2.resize(frame, (224, 224))
        results = model(resized, imgsz=224, verbose=False)

        # Get predicted label
        label = results[0].probs.top1
        class_name = results[0].names[label]

        # Draw the result on original frame
        cv2.putText(frame, f"Posture: {class_name}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Stream frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return '''<html><body><h2>Posture Detection Live</h2><img src="/video_feed"></body></html>'''

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=8554)
    app.run(host='0.0.0.0', port=5000)

import cv2

gst_str = "v4l2src device=/dev/video1 ! video/x-raw, width=640, height=480 ! videoconvert ! appsink"
cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("❌ GStreamer failed to open /dev/video1")
else:
    print("✅ GStreamer capture working — press 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break
        cv2.imshow("Webcam GStreamer", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()