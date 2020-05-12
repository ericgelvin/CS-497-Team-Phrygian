#!flask/bin/python
from flask import Flask, jsonify
from flask import request
import joblib
from sklearn.linear_model import LogisticRegression
import pandas as pd

app = Flask(__name__)

samples = [
    {
        'pregnant':6,
        'glucose': 148,
        'bp':72 ,
        'skin': 35,
        'insulin':0 ,
        'bmi': 33.6,
        'pedigree': 0.627,	
        'age': 50	
    },
    {
        'pregnant':3,
        'glucose': 134,
        'bp':66 ,
        'skin': 38,
        'insulin':0 ,
        'bmi': 32,
        'pedigree': 0.333,	
        'age': 47	
    }
]

@app.route('/data/sample', methods=['GET'])
def get_samples():
    return ' Eric Gelvin\n Assignment 4\n May 8th 2020\n\n'

@app.route('/predict/sample', methods=['POST'])
def predict():
    if not request.json:
        abort(400)
    
    s = {
        'pregnant':request.json['pregnant'],
        'glucose': request.json['glucose'],
        'bp':request.json['bp'] ,
        'insulin':request.json['insulin'] ,
        'bmi': request.json['bmi'],
        'pedigree': request.json['pedigree'],	
        'age': request.json['age']	
        }

    #load the saved model
    model1 = joblib.load('./finalized_model.pkl')
    #predicting
    x = [s['pregnant'],s['glucose'],s['bp'],s['insulin'],s['bmi'],s['pedigree'],s['age']]
    x_test = pd.DataFrame(columns=['pregnant','glucose','bp','insulin','bmi','pedigree','age'])
    x_test.loc[0] = x
    y_pred = model1.predict(x_test)
    return ' Eric Gelvin\n Assignment 4\n May 8th 2020\n result: %s\n\n' % str(y_pred)


if __name__ == '__main__':
    app.run(debug=True)

