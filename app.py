import flask
from flask import Flask, request, jsonify, render_template
import os
import torchxrayvision as xrv
import torch
from PIL import Image
import numpy as np

app = Flask(__name__)
model = xrv.models.DenseNet(weights="all")

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

disease_translations = {
    "Atelectasis": "Ателектаз (спадение доли легкого)",
    "Cardiomegaly": "Кардиомегалия (увеличение размера сердца)",
    "Consolidation": "Консолидация (уплотнение легочной ткани)",
    "Edema": "Отек (накопление жидкости в тканях)",
    "Effusion": "Эффузия (скопление жидкости в плевральной полости)",
    "Emphysema": "Эмфизема (увеличение воздушности легких из-за разрушения альвеол)",
    "Enlarged Cardiomediastinum": "Увеличенный кардиомедиастинальный контур (расширение области средостения)",
    "Fibrosis": "Фиброз (разрастание соединительной ткани в легких)",
    "Fracture": "Перелом (нарушение целостности кости)",
    "Hernia": "Грыжа (выпячивание органа через естественное отверстие)",
    "Infiltration": "Инфильтрация (наполнение легочной ткани клеточными элементами или жидкостью)",
    "Lung Lesion": "Поражение легких (повреждение или изменение ткани)",
    "Lung Opacity": "Затемнение легких (участок непрозрачности на рентгене)",
    "Mass": "Масса (объемное образование в легких)",
    "Nodule": "Узел (небольшое округлое образование в легких)",
    "Pleural_Thickening": "Утолщение плевры (изменение поверхности плевры)",
    "Pneumonia": "Пневмония (воспаление легких)",
    "Pneumothorax": "Пневмоторакс (наличие воздуха в плевральной полости)"
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("upload.html")


@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        img = Image.open(filename).convert('L')
        img = np.array(img.resize((224, 224)))

        img = img / 255.0
        img = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()

        with torch.no_grad():
            preds = model(img)

        results = {pathology: float(pred) for pathology, pred in zip(model.pathologies, preds[0])}

        most_likely_disease = ""
        most_likely_prediction = -1
        for key, value in results.items():
            if value > most_likely_prediction:
                most_likely_prediction = value
                most_likely_disease = key

        translated_most_likely_disease = disease_translations[most_likely_disease]

        response_text = (f"С вероятностью {most_likely_prediction * 100:.2f}% можно "
                         f"диагностировать следующее заболевание: {translated_most_likely_disease}")

        return flask.Response(response_text, content_type='text/plain; charset=utf-8')

    return jsonify({"error": "Invalid file type"}), 400

if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(host="0.0.0.0", port=5001, debug=True) 