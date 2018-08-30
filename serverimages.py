import os
from flask import Flask, render_template, request
from werkzeug import secure_filename
from datetime import datetime

app = Flask(__name__)

UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def name_imgs():
    prefix = 'IMG'
    name = datetime.now().strftime("%Y%m%d_%H%M%S")
    return "{}_{}".format(prefix, name)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['image']
    filename = secure_filename(file.filename)
    _, ext = os.path.splitext(filename)
    name = name_imgs()
    filename = "{}{}".format(name, ext)
    image = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(image)
    return 'OK'
