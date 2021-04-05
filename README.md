# Home-Credit-Default-Risk-Deployment

## Introduction

This repository covers how to deploy a ML model using simple step by step instructions. For this project, I am deploying a basic model for the popular <b>Home
Credit Default Risk</b> Kaggle Problem. Please follow the given link to know more about the data:

https://www.kaggle.com/c/home-credit-default-risk/data

## Business Problem Statement

The main aim is to predict whether a applicant will default on his loan or not. Now to help us with this task, the company has provided a lot of past data about
the applicant like his/her financial history, relevant personal history, previous application details, etc. 

## Machine Learning Problem

Based on the Business problem statement, we can clearly interpret that this is a <b> Binary Classification Problem </b>. The target variable states whether a person
defaults on his/her loan. The value can be interpreted as follows (given in training data):

<ul>
  <li> TARGET 0 : Non-Defaulters </li>
  <li> TARGET 1 : Defaulter </li>
</ul>

The metric we used for the purpose of training the deployment model is <b> AUC Score </b>. This repository just deals with the Deployment part and the actual model training :
for both <i> Deployment </i> and <i> Challenge </i> will be shared in later repositories. For testing this, I have included a Light GBM model pickle file with this repo, located 
in the <i> Model </i> folder. 

## How to test this project 

To test this project, just follow the steps given below:

<ol>
  <li> First, clone this repository in the directory you want your project in. In my case, it was on the virtual server running Ubuntu. </li>
  <li> After that is done, create a new virtual environment. Just run the following commands in the terminal :
        
        sudo apt install python3-venv
        python3 -m venv my-project-env
        source my-project-env/bin/activate
        pip install -r requirements.txt
        
  </li>
  <li> After this is done, just run the app.py file:
        <i> python3 app.py </i> 
       
       If everything goes as planned, your flask app should be up and running on port 5000. You can do the following:
       <ul>
        
        Go to **<your ip or localhost>:5000/fetch_data** to open up a form which takes the applicant data input. You also have to upload 
        a csv file for the previous application data. I have incuded an example csv file in the  uploads/ folder. You only need that specific
        columns. I will add an additional file stating what features you need for the current and previous application data for testing. 
        
        Once you enter all the data and press the *Predict* button, you will be redirected to a new page which just returns a raw json file with 
        predicted label and probablity value.
        
       </ul>
  
  <li> Hence, this is all the project can do for now. </li>
  
  </ol>
  
  ## Future Plans
  
  I do plan to add additonal pages for this project. In particular, a page where a model can be trained too and then tested there itself. Please do support this 
  project by sharing it if you liked it and found it to be useful. 
  
  Thank you!
