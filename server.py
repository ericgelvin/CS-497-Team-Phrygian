#!flask/bin/python
from flask import Flask, jsonify
from flask import request, render_template
import joblib
from sklearn.linear_model import LogisticRegression
import pandas as pd

app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('HomePage.html')


@app.route('/data/sample', methods=['GET'])
def get_samples():
    return ' Eric Gelvin\n Assignment 4\n May 8th 2020\n\n'

@app.route('/predict/sample', methods=['POST'])
def predict():
    
    

    print("WE MADE IT")
    #load the saved model
    model1 = joblib.load('./model.pkl')
    #predicting
    #y_pred = model1.predict()
    return ' Eric Gelvin\n Assignment 4\n May 8th 2020\n result: \n\n' 



if __name__ == '__main__':
    app.run(debug=True)

