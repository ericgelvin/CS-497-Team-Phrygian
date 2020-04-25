#!flask/bin/python
from flask import Flask, jsonify
from flask import request
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

@app.route('/data/sample', methods=['GET'])
def get_samples():
    # https://www.kaggle.com/washingtonpost/police-shootings
    import zipfile
    with zipfile.ZipFile("../input/data/police-shootings.zip","r") as z:
        z.extractall(".")   
    from subprocess import check_output
    print(check_output(["ls", "police-shootings"]).decode("utf8"))
    d = pd.read_csv("police-shootings/police-shootings.csv")
    d.head()
    return jsonify({'samples': d})

@app.route('/predict/sample', methods=['POST'])
def predict():
    if not request.json:
        abort(400)
    
    # TODO: get finalized model or build here
    model1 = joblib.load('./finalized_model.pkl')
    y_pred = model1.predict(X_test)
    return jsonify({'result': y_pred})


if __name__ == '__main__':
    app.run(debug=True)

