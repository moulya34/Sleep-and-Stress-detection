from flask import Flask, render_template, request, flash, redirect
import sqlite3
import pickle
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import pandas as pd
from warnings import filterwarnings

filterwarnings('ignore')


app = Flask(__name__)

# predict_sleep_disorder.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

def predict_sleep_disorder(input_data):
    """
    Predicts sleep disorder using the trained ANN model
    
    Args:
        input_data (list): List of feature values in this order:
            [Gender, Age, Sleep Duration, Quality of Sleep,
            Physical Activity Level, Stress Level, BMI Category, 
            Heart Rate, Daily Steps, ecg, spo2]
            
    Returns:
        str: Predicted sleep disorder
    """
    # Load the saved model and preprocessing objects
    model = load_model('sleep/ANN_model.h5')
    scaler = pickle.load(open('sleep/scaler.pkl', 'rb'))
    le = pickle.load(open('sleep/label_encoder.pkl', 'rb'))
    
    # Convert input to numpy array and reshape
    sample_data = np.array(input_data).reshape(1, -1)
    
    # Scale the features
    sample_data = scaler.transform(sample_data)
    
    # Make prediction
    prediction = model.predict(sample_data)
    predicted_class = np.argmax(prediction, axis=1)
    
    # Decode the prediction
    predicted_disorder = le.inverse_transform(predicted_class)[0]
    
    # Get prediction probabilities
    probabilities = prediction[0]
    class_probabilities = {le.inverse_transform([i])[0]: float(probabilities[i]) 
                          for i in range(len(probabilities))}
    
    return {
        'predicted_disorder': predicted_disorder,
        'probabilities': class_probabilities
    }

    

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')

@app.route('/graph')
def graph():
    names=["confusion_matrix ANN"
    ,"f1_scores ANN",
    "precision_recall_curve ANN","roc_curve ANN"]

    image_path=["http://127.0.0.1:5000/static/confusion_matrix.png",
    "http://127.0.0.1:5000/static/f1_scores.png",
    "http://127.0.0.1:5000/static/precision_recall_curve.png","http://127.0.0.1:5000/static/roc_curve.png"]
    
    return render_template('graph.html',names=names,image_path=image_path,zip=zip)

    
@app.route('/logged')
def logged():
    return render_template('fetal.html')


@app.route('/stress', methods=['GET','POST'])
def stress():
    if request.method == 'POST':
        
        val1=request.form['Age']
        val2=request.form['Gender']
        val3=request.form['Occupation']
        val4=request.form['Marital_Status']
        val5=request.form['Sleep_Duration']
        val6=request.form['Sleep_Quality']
        val7=request.form['Physical_Activity']
        val8=request.form['Screen_Time']
        val9=request.form['Caffeine_Intake']
        val10=request.form['Alcohol_Intake']
        val11=request.form['Smoking_Habit']
        val12=request.form['Work_Hours']
        val13=request.form['Travel_Time']
        val14=request.form['Social_Interactions']
        val15=request.form['Meditation_Practice']
        val16=request.form['Exercise_Type']
        val17=request.form['Blood_Pressure']
        val18=request.form['Cholesterol_Level']
        val19=request.form['Blood_Sugar_Level']
        # Load scaler and model
        scalers = joblib.load('input_scaler.pkl')
        models = load_model('ann_model.h5')
        new_input = np.array([[val1,val2,val3,val4,val5
        ,val6,val7,val8,val9,val10
        ,val11,val12,val13,val14,val15,
        val16,val17,val18,val19]])
        # Scale the input
        new_input_scaled = scalers.transform(new_input)

        # Make prediction
        prediction = models.predict(new_input_scaled)
        predicted_class = np.argmax(prediction, axis=1)[0]
        
        
        
        
        
        print(f"\n\n\n\n {predicted_class} \n\n\n")
        print(f"\n\n\n\n {type(predicted_class)} \n\n\n")
        names=["confusion_matrix ANN"
            ,"f1_scores ANN",
            "precision_recall_curve ANN",
            "roc_curve ANN"]

        image_path=["http://127.0.0.1:5000/static/confusion_matrix.png",
                    "http://127.0.0.1:5000/static/f1_scores.png",
                    "http://127.0.0.1:5000/static/precision_recall_curve.png",
                    "http://127.0.0.1:5000/static/roc_curve.png"]
        return render_template('out.html', prediction=predicted_class,names=names,image_path=image_path,zip=zip)
    
    return render_template('stress.html')


# @app.route('/stress', methods=['GET','POST'])
# def stress():
#     if request.method == 'POST':
#         # Get form data and process
#         features = [float(request.form[col]) for col in [
#             'Age', 'Gender', 'Occupation', 'Marital_Status', 'Sleep_Duration',
#             'Sleep_Quality', 'Physical_Activity', 'Screen_Time', 'Caffeine_Intake',
#             'Alcohol_Intake', 'Smoking_Habit', 'Work_Hours', 'Travel_Time',
#             'Social_Interactions', 'Meditation_Practice', 'Exercise_Type',
#             'Blood_Pressure', 'Cholesterol_Level', 'Blood_Sugar_Level'
#         ]]
#         print(f"\n\n\n\n{features}\n\n\n")
#         agee=int(request.form['Age'])
#         # Load scaler and model
#         scaler = joblib.load('input_scaler.pkl')
#         model = load_model('ann_model.h5')
#         if agee == 25:
#             print("\n\n\n entered \n\n\n\n")
#             predicted_class = 0
#         else:
#             # Convert input to DataFrame without column names
#             scaled_input = scaler.transform(pd.DataFrame([features]))
#             # scaled_input = scaler.transform(pd.DataFrame(features))
#             prediction = model.predict(scaled_input)
#             predicted_class = np.argmax(prediction, axis=1)[0]

#         names=["confusion_matrix ANN"
#             ,"f1_scores ANN",
#             "precision_recall_curve ANN",
#             "roc_curve ANN"]

#         image_path=["http://127.0.0.1:5000/static/confusion_matrix.png",
#                     "http://127.0.0.1:5000/static/f1_scores.png",
#                     "http://127.0.0.1:5000/static/precision_recall_curve.png",
#                     "http://127.0.0.1:5000/static/roc_curve.png"]
                
    
    
#         return render_template('out.html', prediction=predicted_class,names=names,image_path=image_path,zip=zip)
#     return render_template('stress.html')

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM user WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchall()

        if result:
            return render_template('stress.html')
        else:
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided,  Try Again')

    return render_template('index.html')


@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
        cursor.execute(command)

        cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()

        return render_template('index.html', msg='Successfully Registered')
    
    return render_template('index.html')

@app.route('/logout')
def logout():
    return render_template('index.html')


@app.route("/fetalPage", methods=['GET', 'POST'])
def fetalPage():
    return render_template('fetal.html')




@app.route("/predict", methods = ['POST', 'GET'])
def predictPage():
    if request.method == 'POST':
        name = request.form['name']
        gender = request.form['gender']
        age = request.form['age']
        slp_d = request.form['slp_d']
        qos = request.form['qos']
        pal = request.form['pal']
        sl = request.form['sl']
        bmi = request.form['bmi']
        hr = request.form['hr']
        ds = request.form['ds']
        ecg = request.form['ecg']
        spo2 = request.form['spo2']
        
        sample_input=[gender, age, slp_d, qos, pal, sl, bmi,hr,ds,ecg,spo2]
        # Make prediction
        result = predict_sleep_disorder(sample_input)
        
        # Print results
        print("\nSleep Disorder Prediction Results")
        print("="*40)
        print(f"Input Features: {sample_input}")
        print(f"\nPredicted Sleep Disorder: {result['predicted_disorder']}")
        print("\nPrediction Probabilities:")
        for disorder, prob in result['probabilities'].items():
            print(f"{disorder}: {prob:.4f}")

        res=result['predicted_disorder']
        names=["confusion_matrix graph ANN"
            ,"f1_scores graph ANN",
            "precision_recall_curve graph ANN",
            "roc_curve graph ANN"]

        image_path=["http://127.0.0.1:5000/static/cm.png",
                    "http://127.0.0.1:5000/static/f1.png",
                    "http://127.0.0.1:5000/static/pr.png",
                    "http://127.0.0.1:5000/static/roc.png"]
                
                
        print(res)

           
        return render_template('predict.html',name=name, pred = res,status=res,names=names,image_path=image_path,zip=zip)

    return render_template('predict.html')

if __name__ == '__main__':
	app.run(debug = True)