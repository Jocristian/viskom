import sys
import os
from flask import Flask, render_template, Response, jsonify

# Tambahkan path ke folder 'src'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'face_classification', 'src'))

from video_camera import VideoCamera

app = Flask(__name__)  # HARUS didefinisikan dulu

camera = VideoCamera()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def gen(camera):
        while True:
            frame = camera.get_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(gen(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect_emotion')
def detect_emotion():
    emotion = camera.get_last_detected_emotion()
    return jsonify({'emotion': emotion})

if __name__ == "__main__":
    app.run(debug=True)
