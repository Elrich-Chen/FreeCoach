from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
import os
from werkzeug.utils import secure_filename
from utilities.pose_predict import predict_video

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
app.config['SECRET_KEY']= "auwsdeyjfvjueryuebu"

filename = ""

# Ensure the uploads folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Helper function to check file type
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Routes
@app.route('/')
def index():
    return render_template('index.html', video = filename)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return redirect(url_for('index'))
    video = request.files['video']
    if video.filename == '':
        return redirect(url_for('index'))
    if video and allowed_file(video.filename):
        filename = secure_filename(video.filename)
        video.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        prediction, confidence = predict_video(f"uploads/{filename}")
        print(f"\n\n\n\n\n\n\nPrediction: {prediction}")
        print(f"Confidence: {confidence:.2f}")
        flash(f"Prediction: {prediction}")
        flash(f"Confidence: {confidence:.2f}")
        return render_template('index.html', video = filename, predict_video = "predict.mp4")

    return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
