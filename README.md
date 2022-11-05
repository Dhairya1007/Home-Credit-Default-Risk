# Home Credit Default Risk

# ![Home-Credit-logo](https://user-images.githubusercontent.com/33721658/200103652-bf39b7d6-21e1-49be-ac55-cbe1835291a0.jpg)

### Introduction

This section covers an end-to-end solution approarch for the popular <b>Home
Credit Default Risk</b> Kaggle Problem. 

Please go to this like to get more information about this problem.
https://www.kaggle.com/c/home-credit-default-risk/data


### Business Problem Statement

The main aim is to predict whether an applicant will default on his loan or not. Now to help us with this task, the company has provided a lot of past data about
the applicant like his/her financial history, relevant personal history, previous application details, etc. 

## Machine Learning Problem

Based on the Business problem statement, we can clearly interpret that this is a <b> Binary Classification Problem </b>. The target variable states whether a person
defaults on his/her loan. The value can be interpreted as follows (given in training data):

<ul>
  <li> TARGET 0 : Non-Defaulters </li>
  <li> TARGET 1 : Defaulter </li>
</ul>

The metric we used for the purpose of training the deployment model is <b> AUC Score </b>. 

## Solution

The solution for this problem can be broken into 4 parts for easier understanding: 

1. <b>Exploratory Data Analysis</b> - This is covered in the Home_Credit_EDA notebook. The main goal is to understand the data to make some useful inferences.

2. <b>Data Preprocessing and Feature Engineering</b> - Once we have a sense of what the data is like, we will use the resultant inferences from the first section to process our data (cleaning, handling missing data and outliers, etc) and create some useful new features using the processed data. This is covered in the first two sections of the Home_Credit_Modelling notebook.

3. <b>Data Modelling</b> - Once we have the train and test data ready, we will start with data modelling and assess the performance of various popular machine learning models. This has been covered in the Home_Credit_Modelling notebook.

4. <b>Model Deployment</b> - This is the last section in which we will test a lighter version of our model by actually deploying it as an application. We will cover how to setup this on any server of your choice and how to test this as well.

Please note that you will find detailed comments on each step for both the EDA and Modelling notebooks (placed in the notebooks folder). Please feel free to connect with me incase you have any doubts on dhairyaparikh1998@gmail.com

Now, we will cover do you deploy and test a created model through a deplyoed webpage.

## Project Deployment

### How to test this project 

To test this project, just follow the steps given below:

<ol>
  <li> First, clone this repository in the directory you want your project in. In my case, it was on the virtual server running Ubuntu. </li>
  <li> After that is done, create a new virtual environment. Just run the following commands in the terminal :
        
        sudo apt install python3-venv
        python3 -m venv my-project-env
        source my-project-env/bin/activate
        pip install -r requirements.txt
        
  </li>
  <li> After this is done, just run the app.py file: </li>
        
  <i> python3 app.py </i> 
</ol>      
  
  If everything goes as planned, your flask app should be up and running on port 5000. You can do the following:
       
  Go to **--IP Address--:5000/fetch_data** to open up a form which takes the applicant data input. You also have to upload 
  a csv file for the previous application data. I have incuded an example csv file in the  uploads/ folder. You only need that specific
  columns. I will add an additional file stating what features you need for the current and previous application data for testing. 
        
  Once you enter all the data and press the *Predict* button, you will be redirected to a new page which just returns a raw json file with 
  predicted label and probablity value.
       
  
## Future Plans
  
  I do plan to add additonal pages for this project. In particular, a page where a model can be trained too and then tested there itself. Please do support this 
  project by sharing it if you liked it and found it to be useful. 
  
  Thank you!
