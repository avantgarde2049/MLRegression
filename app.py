import pickle
from flask import render_template,url_for,jsonify,app,request,Flask
import numpy as np
import pandas as pd

app=Flask(__name__)
##Load the model and the scaler it was trained with
regmodel=pickle.load(open('regmodel.pkl','rb'))
scaler=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def homepage():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
   data=request.json['data']
   print(data)
   # DataFrame keeps feature names, so the scaler matches columns by name
   new_data=scaler.transform(pd.DataFrame([data])[scaler.feature_names_in_])
   output=regmodel.predict(new_data)
   print(output[0])
   return jsonify(float(output[0]))


if __name__=="__main__":
    app.run(debug=True)