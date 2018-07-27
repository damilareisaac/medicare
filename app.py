import os
from flask import Flask, flash, redirect, render_template, request, session, url_for
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression
import sqlite3 as sql


heart_clf = joblib.load('pkl_objects/heart_classifier.pkl')
app = Flask(__name__)


def predict_disease(form_data):
    form_data = dict(form_data)
    data = pd.read_csv('data/data_pivoted.csv')
    data = data.fillna(0)
    columns = ['abdomen acute', 'abnormally hard consistency', 'abscess bacterial',
               'agitation', 'angina pectoris', 'apyrexial', 'arthralgia', 'asthenia',
               'breath sounds decreased', 'cardiovascular event', 'chest tightness',
               'cough', 'decreased body weight', 'diarrhea', 'difficulty passing urine',
               'distress respiratory', 'dizziness', 'drowsiness', 'facial paresis',
               'fatigue', 'fever', 'haemorrhage', 'muscle twitch', 'nausea', 'neck stiffness',
               'pain', 'pain abdominal', 'pain chest', 'paralyse', 'pin-point pupils',
               'pleuritic pain', 'seizure', 'shortness of breath', 'sleeplessness', 'stiffness',
               'swelling', 'tremor', 'unresponsiveness', 'vomiting', 'wheezing', 'worry', 'yellow sputum']
    symptoms = data[columns]
    disease = data.disease
    model = LogisticRegression()
    model = model.fit(symptoms, disease)
    del(form_data['disease_symptom'])
    if len(form_data) == 0:
        result = 'No symptoms selected. Please select one or more symptoms'
    else:
        data = OrderedDict()
        for key in form_data:
            value = float(form_data[key][0])
            data[key] = value
        data = OrderedDict()
        for i in columns:
            if i in form_data.keys():
                data[i] = 1.0
            else:
                data[i] = 0.0
        input_data = list(data.values())
        input_data = np.array(input_data)
        input_data = input_data.reshape(1, -1)
        result = list(model.predict(input_data))[0]
        result = "Disease predicted is '{0}'".format(result)
    return result


def predict_heart(form_data):
    data = OrderedDict()
    for key in form_data:
        values = float(form_data[key])
        data[key] = values
    if data['fasting_blood_sugar'] > 120:
        data['fasting_blood_sugar'] = 1
    else:
        data['fasting_blood_sugar'] = 0
    data = list(data.values())
    data = np.array(data)
    data = data.reshape(1, -1)
    prediction = heart_clf.predict(data)[0]
    label = {0: 'No heart Disease', 1: 'Heart Disease Diagnosed base on answers'}
    resfinal = label[prediction]
    return resfinal


def calculate_bmi(form):
    bmi = 0.00
    if form['height_type'] == 'm' and form['weight_type'] == 'kg':
        height_m = float(form['height'])
        weight_kg = float(form['weight'])
        bmi = weight_kg / (height_m * height_m)

    if form['height_type'] == 'm' and form['weight_type'] == 'lb':
        height_m = float(form['height'])
        weight_lb = float(form['weight'])
        bmi = (weight_lb * 0.45359237) / (height_m * height_m)

    if form['height_type'] == 'in' and form['weight_type'] == 'kg':
        height_in = float(form['height'])
        height_in = 0.0254 * height_in
        weight_kg = float(form['weight'])
        bmi = weight_kg / (height_in * height_in)

    if form['height_type'] == 'in' and form['weight_type'] == 'lb':
        height_in = float(form['height'])
        height_in = 0.0254 * height_in
        weight_lb = float(form['weight'])
        bmi = (weight_lb * 0.45359237) / (height_in * height_in)

    if bmi < 18.5:
        bmi_status = """You are Underweight. <br>
         Foods that should be included every day: <br>
        1. Full-cream milk: 750 - 1000 ml (3 to 4 cups) <br>
        2. Meat, fish, eggs and other protein foods: 3-5 servings (90 to 150 g). <br>
        3. Bread and cereals: 8-12 servings (e.g. up to 6 cups of starch a day) <br>
        4. Fruit and vegetables: 3-5 servings. <br>
        5. Fats and oils: 90 g (6 tablespoons. <br>
        6. Healthy desserts: 1-2 servings. <br>"""
    elif bmi < 24.9:
        bmi_status = """You are Normal. Current diet plan is ok."""
    elif bmi < 29.9:
        bmi_status = """You are Overweight.<br>
        1. Increase The Consumption Of Fruits And Vegetables. <br>
        2. Limit The Intake Of Stimulants such as caffeine, alcohol, and refined sugar.<br>
        3. Do Not Skip Breakfast as is the most important meal of the day. <br>
        4. Drink Plenty Of Water. <br>
        5. Have Smaller Gaps Between The Meals. <br>
        6. Do Not Starve. <br>
        7. Restrict Your Calorie Intake. <br>
        8. Remove Fat From Your Food. <br>
        9. Eat Healthy Snacks. <br>"""
    else:
        bmi_status = """You are Obese. Diet plan as follows:<br>
            1. Cut your caloric intake by about 500 to 1,000 calories a day.<br>
            2. Restrict carbohydrates -- particularly high-glycemic varieties that affect your blood sugar. <br>
            3. Cut back on portions to eat less food and balance your caloric intake.<br>
            4. Combine Diet with Exercise. <br>"""
    bmi_status = 'Your Body Mass Index is {0:1.2f}, {1}'.format(bmi, bmi_status)
    return bmi_status


@app.route('/', methods=['POST', 'GET'])
def home():
    bmi_status = ''
    resfinal = ''
    disease = ''
    if session.get('logged_in'):
        if request.method == 'POST':
            form = request.form
            if 'disease_symptom' in form.keys():
                disease = predict_disease(form_data=form)

            if 'height' in form.keys():
                bmi_status = calculate_bmi(form=form)

            if 'age' in form.keys():
                resfinal = predict_heart(form)
        return render_template('home.html', bmi_status=bmi_status, heart_status=resfinal, pred_disease=disease)
    return redirect(url_for('login'))


@app.route('/login', methods=['POST', 'GET'])
def login():
    msg= "new"
    if session.get('logged_in'):
        return redirect(url_for('home'))
    if request.method == 'POST':
        form = request.form
        try:
            with sql.connect("database.db") as con:
                cur = con.cursor()
                cur.execute("SELECT username, password from Account")
                for username, password in cur:
                    if form['username'] == username and form['password'] == password:
                        msg = "success"
                        session['logged_in'] = True
                        return redirect(url_for('home'))
                    pass
                else:
                    msg = "invalid login"
        except:
            con.rollback()
            msg = "error"
    return render_template('login.html', msg=msg)


@app.route('/register', methods = ['POST', 'GET'])
def register():
    msg = "new"
    if request.method == 'POST':
        form = request.form
        username = form['username']
        password = form['password']
        confirm_password = form['confirm password']

        try:
            with sql.connect("database.db") as con:
                cur = con.cursor()
                cur.execute("SELECT username, password from Account")
                for registered_user, password in cur:
                    if registered_user == username:
                        msg = "username already exist"
        except:
            con.rollback()
            msg = "error"
        finally:
            con.close()

        if password != confirm_password:
            msg = "mismatch password"

        if msg != "username already exist" and msg != "mismatch password" :
            try:
                with sql.connect("database.db") as con:
                    cur = con.cursor()
                    cur.execute("INSERT INTO Account "
                            "(username, password) VALUES (?, ?)", (username, password))
                    con.commit()
                    msg = "success"
                    session['logged_in'] = True
                    return redirect(url_for('home'))
                pass
            except:
                con.rollback()
                msg = "error"
            finally:
                return redirect(url_for('home'))
                con.close()
    return render_template("register.html", msg=msg)

@app.route('/logout')
def logout():
    session['logged_in'] = False
    return redirect(url_for('login'))




if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(debug=True)
