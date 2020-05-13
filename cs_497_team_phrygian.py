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
    d = pd.read_csv("./database.csv")
    d.head()
    return jsonify({'samples': d})

@app.route('/predict/sample', methods=['POST'])
def predict():
    if not request.json:
        abort(400)

    s = {
        'manner_of_death':request.json['manner_of_death'],
        'armed': request.json['armed'],
        'age':request.json['age'] ,
        'gender':request.json['gender'] ,
        'race': request.json['race'],
        'city': request.json['city'],	
        'state': request.json['state'],
        'signs_of_mental_illness': request.json['signs_of_mental_illness'],
        'threat_level': request.json['threat_level'],
        'flee': request.json['flee'],
        'body_camera': request.json['body_camera']
        }

    #load the saved model
    model1 = joblib.load('./model.pkl')
    #predicting
    x = [s['manner_of_death'],s['armed'],s['age'],s['gender'],s['race'],s['city'],s['state'],s['signs_of_mental_illness'],s['threat_level'],s['flee'],s['body_camera']]
    x_test = pd.DataFrame(columns=['manner_of_death','armed','age','gender','race','city','state','signs_of_mental_illness','threat_level','flee','body_camera'])
    x_test.loc[0] = x
    y_pred = model1.predict(x_test)
    return ' result: %s\n\n' % str(y_pred)

if __name__ == '__main__':
    app.run(debug=True)

