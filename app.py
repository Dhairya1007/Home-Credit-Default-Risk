from flask import Flask, flash, render_template, request, jsonify
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import lightgbm as lgbm
import matplotlib.pyplot as plt
import os
from time import time
from scipy.sparse import hstack

# Sklearn Imports
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

# Function for processing the form input data
def process(app_data, prev_data):
      '''
      This function preprocesses the loaded data. The processing includes:

        4. Feature Engineering to create some better features

      Inputs:
        app_data : application data
        prev_data : previous applications data
      
      Returns:
        final_datapoint : final_datapoint for testing
      '''
    
      # --------------------------------------------------------------------------------------------------------------

      # Task : Feature Engineering
      # --------------------------------------------------------------------------------------------------------------
      
      # i) Application_train/test Feature Engineering

      app_data = pd.Series(app_data)
      
      app_data['DAYS_ID_PUBLISH'] = - int(app_data['DAYS_ID_PUBLISH'])
      app_data['NEW_CREDIT_TO_ANNUITY_RATIO'] = float(app_data['AMT_CREDIT']) / float(app_data['AMT_ANNUITY'])
      app_data['NEW_CREDIT_TO_GOODS_RATIO'] = float(app_data['AMT_CREDIT']) / float(app_data['AMT_GOODS_PRICE'])
      app_data['INCOME_PER_PERSON'] = float(app_data['AMT_INCOME_TOTAL']) / float(app_data['CNT_FAMILY_MEMBERS'])
      app_data['ANNUITY_INCOME_PERC'] = float(app_data['AMT_ANNUITY']) / float(app_data['AMT_INCOME_TOTAL'])
      app_data['INCOME_CREDIT_PERC'] = float(app_data['AMT_INCOME_TOTAL']) / float(app_data['AMT_CREDIT'])
      app_data['DAYS_EMPLOYED_PERC'] = float(app_data['YEARS_EMPLOYED']) / float(app_data['AGE'])
      app_data['PAYMENT_RATE'] = float(app_data['AMT_ANNUITY']) / float(app_data['AMT_CREDIT'])

      # Handling Categorical Data
      # 1. Gender
      if app_data['CODE_GENDER'] == 'F':
        app_data['CODE_GENDER_F'] = 1
        app_data['CODE_GENDER_M'] = 0
        app_data['CODE_GENDER_nan'] = 0
        app_data.drop(labels = ['CODE_GENDER'], inplace = True)

      elif app_data['CODE_GENDER'] == 'M':
        app_data['CODE_GENDER_F'] = 0
        app_data['CODE_GENDER_M'] = 1
        app_data['CODE_GENDER_nan'] = 0
        app_data.drop(labels = ['CODE_GENDER'], inplace = True)
      
      else:
        app_data['CODE_GENDER_F'] = 0
        app_data['CODE_GENDER_M'] = 0
        app_data['CODE_GENDER_nan'] = 1
        app_data.drop(labels = ['CODE_GENDER'], inplace = True)

      # 2. FLAG_OWN_CAR
      if app_data['FLAG_OWN_CAR'] == 'N':
        app_data['FLAG_OWN_CAR_N'] = 1
        app_data['FLAG_OWN_CAR_Y'] = 0
        app_data['FLAG_OWN_CAR_nan'] = 0
        app_data.drop(labels = ['FLAG_OWN_CAR'], inplace = True)
      
      elif app_data['FLAG_OWN_CAR'] == 'Y':
        app_data['FLAG_OWN_CAR_N'] = 0
        app_data['FLAG_OWN_CAR_Y'] = 1
        app_data['FLAG_OWN_CAR_nan'] = 0
        app_data.drop(labels = ['FLAG_OWN_CAR'], inplace = True)

      else:
        app_data['FLAG_OWN_CAR_N'] = 0
        app_data['FLAG_OWN_CAR_Y'] = 1
        app_data['FLAG_OWN_CAR_nan'] = 0
        app_data.drop(labels = ['FLAG_OWN_CAR'], inplace = True)


      # ii) Previous Application Data
      prev_data['APP_CREDIT_PERC'] = prev_data['AMT_APPLICATION'] / prev_data['AMT_CREDIT']

      # Handling Categorical Data : Only One Cat column : NAME_CONTRACT_STATUS
      prev_data['NAME_CONTRACT_STATUS_Approved'] = np.where(prev_data['NAME_CONTRACT_STATUS'] == 'Approved', 1, 0)
      prev_data['NAME_CONTRACT_STATUS_Refused'] = np.where(prev_data['NAME_CONTRACT_STATUS'] == 'Refused', 1, 0)
      prev_data['NAME_CONTRACT_STATUS_nan'] = np.where(prev_data['NAME_CONTRACT_STATUS'] == np.nan, 1, 0)
      prev_data.drop(columns = ['NAME_CONTRACT_STATUS'], inplace= True)

      # Aggregations we will perform 
      prev_app_num_agg = {
                      'AMT_ANNUITY': ['min', 'max', 'mean'],
                      'AMT_APPLICATION': ['min', 'max', 'mean'],
                      'AMT_CREDIT': ['min', 'max', 'mean'],
                      'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
                      'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
                      'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
                      'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
                      'DAYS_DECISION': ['min', 'max', 'mean'],
                      'CNT_PAYMENT': ['mean', 'sum']
                    }
      
      prev_data['ID'] = 1

      # Perform the aggregations based on the sk_id_curr
      prev_agg_features = prev_data.groupby('ID').agg(prev_app_num_agg)
      prev_agg_features.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg_features.columns.tolist()])

      # Previous Applications: Approved Applications - only numerical features
      approved_applications = prev_data[prev_data['NAME_CONTRACT_STATUS_Approved'] == 1]
      approved_agg_features = approved_applications.groupby('ID').agg(prev_app_num_agg)
      approved_agg_features.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg_features.columns.tolist()])
      prev_agg_features = prev_agg_features.join(approved_agg_features, how='left', on='ID')
      
      # Previous Applications: Refused Applications - only numerical features
      refused_applications = prev_data[prev_data['NAME_CONTRACT_STATUS_Refused'] == 1]
      refused_agg_features = refused_applications.groupby('ID').agg(prev_app_num_agg)
      refused_agg_features.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg_features.columns.tolist()])
      prev_agg_features = prev_agg_features.join(refused_agg_features, how='left', on='ID')

      final_data = pd.concat([app_data,prev_agg_features.iloc[0]], axis = 0)

      return final_data

# Function for predicting the value

def predict_val(datapoint):

   # Loading the model
   model = pd.read_pickle('Model/lgbm_deployment_model.pkl')

   # Getting the prediction probability
   start_time = time()

   best_model_thres = 0.0834754

   pred_value = model.predict_proba(datapoint)[:,1]
   predicted_label = np.where(pred_value > best_model_thres, 1, 0)

   print("-" * 100)
   print(f"Predicted Probabilties for given Client(s) being Defaulter is/are: {np.round(pred_value, 4)}")
   print(f"The class label for given query point is: {predicted_label}")
   print(f"Total Time Taken for prediction = {round(time() - start_time, 2)}"+ " seconds.")
   print('-' * 100)

   return pred_value, predicted_label

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------------------- Main Code for Flask --------------------------------------------
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['txt', 'csv'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def hello_world():
   return 'Home Credit Defualt Risk Deployment Example'

@app.route('/fetch_data')
def get_data():
    return render_template('index.html')
	
@app.route('/predict', methods = ['POST'])
def predict():
   if request.method == 'POST':
       # Fetch all the user data into variables
       
       app_data = request.form.to_dict()
       del app_data['uname']

       # check if the post request has the file part
       file = request.files['prev_app_data']

       if file.filename == '':
          print('No file selected for uploading')
          

       if file and allowed_file(file.filename):
          filename = secure_filename(file.filename)
          file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
          print('File successfully uploaded')
          
       else:
          print('Allowed file types csv and txt only.')
          
        
       prev_data = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
       
       final_dp = process(app_data, prev_data)
       x = np.array(final_dp).reshape(1,104)
       prediction_prob, pred_label = predict_val(x)

   return {'Predicted Probability' : float(prediction_prob), 'Predicted_Label' : int(pred_label)}
		
if __name__ == '__main__':
   app.run(debug = True)