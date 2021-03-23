import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl","rb"))
model2 = pickle.load(open("model2.pkl","rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    features = [int(x) for x in request.form.values()]

    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    prediction2 = model2.predict(final_features)
    print(str(prediction))
    output='{0:.{1}f}'.format(prediction[0][0], 2)
    output2='{0:.{1}f}'.format(prediction2[0][0], 2)
    Toplam=float(output)+float(output2)

    return render_template('index.html',
                           pred='Sag Akciğer Hacmi {} // Sol Akciğer Hacmi {} // Toplam Akciğer Hacmi {}'.format(str(float(output)),str(float(output2)),str(Toplam)))


if __name__ == "__main__":
    app.run(debug=True)
    