import os
from flask import Flask, render_template, request
from flask import jsonify,json
from flask_cors import CORS
from werkzeug.utils import secure_filename
from datetime import datetime
from PretrainedModels.pretrained_cnns import PredictTop5
app = Flask(__name__)

UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

CORS(app, expose_headers='Authorization')

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
    results = PredictTop5(image_path=image, model_name='VGG19')
    jsonStr = json.dumps(results)
    return jsonStr
