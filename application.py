import pickle
from flask import Flask,request,render_template, url_for
import numpy as np
import pandas as pd
from src.logger import logging
import webbrowser
import threading

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application = Flask(__name__)
app = application

#Route for a Home Page

@app.route('/')
def index():
    return render_template('index.html')
def open_browser():
    webbrowser.open_new('http://localhost:5000/')

@app.route('/about_gdm')
def about_gdm():
    return render_template('about_gdm.html')

@app.route('/faq')
def faq():
    return render_template('faq.html')

@app.route('/contact_us')
def contact_us():
    return render_template('contact_us.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('prediction.html',results=None)
    else:
        weight = request.form.get('Weight')
        height = request.form.get('Height')
        bmi = request.form.get('BMI')
        #If BMI not provided
        if not bmi and weight and height:
            # Convert weight and height to floats and calculate BMI
            bmi = (float(weight)*10000) / (float(height) ** 2)

        bmi = float(bmi)  #To ensure it ain't missing

        data=CustomData(
            BMI= bmi,
            Dia_BP= float(request.form.get('Dia_BP')),
            OGTT= float(request.form.get('OGTT')),
            PCOS= request.form.get('PCOS'),
            Prediabetes= request.form.get('Prediabetes')
        )

        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        if(results[0]==0.0):
            msg = "No"
        else:
            msg = "Yes"
        logging.info(f"Predicted result: {results[0]}, Message: {msg}")
        return render_template('post.html',results=msg)


if __name__=="__main__":
    threading.Timer(1.25, open_browser).start()
    app.run(host="0.0.0.0", port=5000)
