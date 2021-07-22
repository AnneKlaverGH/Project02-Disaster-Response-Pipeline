# Disaster Response Pipeline Project

## Project Motivation
In this project, we analyzed disaster data from Figure Eight to build a model for an API that classifies disaster messages. We have used a data set containing real messages that were sent during disaster events. We created a machine learning pipeline to categorize these events so that we can send the messages to an appropriate disaster relief agency. The project also includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data.

## File Descriptions
* run.py - python script for web application
* go.html - html script for web application
* master.html - html script for web application

* disaster_messages.csv - messages data set
* disaster_categories.csv - categories of the messages
* process_data.py - python code to prepare data en store into SQLite database
* DiasterResponse.db - SQLite database

* train_classifier.py - python code to train model and store model into pickle file
* classifier.pkl - contains the (trained) model

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
