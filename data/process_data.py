# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Loads data from csv to df

    Args:
    messages_filepath: file path to disaster_messages.csv
    categories_filepath: file path to disaster_categories.csv

    Returns:
    df: merged dataframe
    """
    
    #load messages dataset
    messages = pd.read_csv(messages_filepath)
    messages.head()
    
    #load categories dataset
    categories = pd.read_csv(categories_filepath)
    categories.head()
    
    # merge datasets
    df = messages.merge(categories, how='left', left_on='id', right_on='id')
    
    return df


def clean_data(df):
    """Function that cleans the df.

    Args:
    df: df that needs to be cleaned

    Returns:
    df: cleaned df
    """
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe and use this row to extract a list of new column names for categories
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x:  x.split('-')[0])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    #Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x:  x.split('-')[1])
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    #Replace categories column in df with new category columns:
    #1. drop the original categories column from `df`
    df = df.drop(['categories'], axis=1)
    #2. concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    #Remove duplicates
    df = df[~df.duplicated()]

    #Remove nans 
    df.dropna(subset=['request'], inplace=True)
    
    return df
    
def save_data(df, database_filename):
    """Stores the clean data into a SQLite database in the specified database file path
    
    Args:
    df: df that needs to be stored
    database_filename: database file path

    Returns:
    None: merged dataframe is stored into a SQLite database
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('final_df', engine, index=False,  if_exists='replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()