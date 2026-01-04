from flask import Flask, render_template,request
from src.pipeline import predict_pipeline, user_input_data

app=Flask(__name__)

@app.route("/home")
def welcome():
    return render_template('index.html')