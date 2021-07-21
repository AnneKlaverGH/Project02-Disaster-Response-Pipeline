# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import nltk
nltk.download(['stopwords', 'punkt','wordnet'])
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import pickle

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+' 
stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()


def load_data(database_filepath):
    """Imports data from SQLite database

    Args:
    database_filepath: database file path

    Returns:
    X: input variable (messages)
    y: output variable (categories)
    category_names: list of category names
    """
    
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('final_df', engine)  
    X = df['message']
    y = df.drop(['id','message','original','genre'], axis = 1)
    category_names = y.columns
    
    return X, y, category_names


def tokenize(text):
    """Custom tokenize function using nltk to case normalize, remove punctuation, lemmatize, tokenize text, and remove stop words. 

    Args:
    text: text to tokenize

    Returns:
    tokens: tokenized text
    """
        
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word).strip() for word in tokens if word not in stop_words]

    return tokens


def build_model():
    """Builds a pipeline that processes text and then performs multi-output classification on the 36 categories in the dataset. 
       GridSearchCV is used to find the best parameters for the model.

    Args:
    None

    Returns:
    cv: GridSearchCV including pipeline
    """

    #build pipeline
    pipeline = Pipeline([
            ('features', FeatureUnion([
                ('text_pipeline', Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer())
                ]))
            ])),
            ('clf', MultiOutputClassifier(RandomForestClassifier()))
            ])
    
    #Gridsearch
    #Takes a while to run (-> reduce the number of parameters to check if code is working properly)
    parameters = {
        'clf__estimator__n_estimators': [10, 20, 50],
        'clf__estimator__min_samples_split': [2, 5, 10],
        'clf__estimator__min_samples_leaf' : [1, 2, 4],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv
    


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluates the model and prints the f1 score, precision and recall for the test set for each category.

    Args:
    model: ml model
    X_test: test input variable (messages)
    Y_test: corresponding categories for the messages in X_test
    category_names: list of category names

    Returns:
    None: prints a classification_report
    """
    
    #predict on test data
    Y_pred = model.predict(X_test)
    
    print(classification_report(np.hstack(Y_test.values), np.hstack(Y_pred), target_names=category_names))


def save_model(model, model_filepath):
    """Stores the classifier into a pickle file to the specified model file path.

    Args:
    model: model that needs to be stored
    model_filepath: file path where the model should be stored

    Returns:
    None: model is stored into a pickle file to the specified file path
    """
    with open(model_filepath, 'wb') as file:  
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()