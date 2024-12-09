# # importing Flask and other modules
# from flask import Flask, request, render_template
# from flask_bootstrap import Bootstrap
# # Flask constructor
# app = Flask(__name__)
# Bootstrap(app)

# # A decorator used to tell the application
# # which URL is associated function
# @app.route('/', methods =["GET", "POST"])
# def gfg():
# 	if request.method == "POST":
# 		age= request.form.get("age")
# 		bp = request.form.get("bp")
# 		return age+ " " + bp
# 	return render_template("main.html")
# if __name__=='__main__':
# 	app.run()


from json import load

import numpy as np
import pandas as pd
import seaborn as sns
import joblib
from flask import Flask, render_template, request, send_file
from flask_bootstrap import Bootstrap
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename

sns.set_style('whitegrid')
# import plotly.express as px
import warnings
from itertools import cycle

warnings.filterwarnings("ignore")
import pickle

# from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# df = pd.read_csv("Heart_Disease_Prediction.csv")    
# dtclas=DecisionTreeClassifier()
# X= df.drop(['Heart Disease'], axis=1)
# y= df['Heart Disease']
# X_train, X_test,y_train, y_test=train_test_split(X,y,test_size=0.25,random_state=40)
# modeldt=dtclas.fit(X_train,y_train)
# pickle.dump(modeldt, open('heart_disease_detector.pkl', 'wb'))


app = Flask(__name__)
Bootstrap(app)
cors = CORS(app)

@app.route('/', methods= ['GET'])
def home():
	return render_template('main.html')

@app.route("/predict", methods =["GET", "POST"])
@cross_origin()
def gfg():   
    
	mod = joblib.load('heart_disease_detector.pkl')
	features = [float(i) for i in request.form.values()]
	print(features)
	array_features = [np.array(features)]
	prediction = mod.predict(array_features)   
	output = prediction
	print(output[0])
	if output == 'Absence':
		return render_template('main.html', result = 'The patient is not likely to have heart disease!')
	else:
		return render_template('main.html', result = 'The patient is likely to have heart disease!')

if __name__=='__main__':
	app.run()