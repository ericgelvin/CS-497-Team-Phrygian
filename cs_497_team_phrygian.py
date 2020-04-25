#!flask/bin/python
from flask import Flask, jsonify
from flask import request
from sklearn.linear_model import LogisticRegression
import joblib

# https://www.kaggle.com/washingtonpost/police-shootings
# TODO: parse kaggle data and split into sample data


app = Flask(__name__)

samples = [
    {
        'name': 'john smith',
        'date': '2020-01-03,
        'manner_of_death': 4,
        'armed': 0,
        'age': 25,
        'gender': 1,
        'race': 3,	
        'city': 'san diego',
        'signs_of_mental_illness': 0,
        'threat_level': 1,
        'flee': 0,
        'body_camera': 0
    }
]

@app.route('/data/sample', methods=['GET'])
def get_samples():
    return jsonify({'samples': samples})

@app.route('/predict/sample', methods=['POST'])
def predict():
    if not request.json:
        abort(400)
    
    sample = {
        }

    # TODO: get finalized model or build here

    #load the saved model
    model1 = joblib.load('./finalized_model.pkl')
    #predicting
    y_pred = model1.predict(X_test)
    return jsonify({'result': y_pred})


if __name__ == '__main__':
    app.run(debug=True)

