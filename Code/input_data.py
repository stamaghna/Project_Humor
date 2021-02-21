import pandas as pd
import os
from sklearn.model_selection import train_test_split

def get_data ( path ):
    df = pd.read_csv(path)
    return df

#Assumption: Text field cannot be not null.
def fill_na ( df ):
    df ['is_humor'] = df['is_humor'].fillna(0)
    df ['humor_rating'] = df['humor_rating'].fillna(0)
    df ['humor_controversy'] = df['humor_controversy'].fillna(0)
    print ('Hi')
    #return df
    #df ['offense_rating'] = df['offense_rating'].fillna(df['offense_rating'].mean())
    #df_1 = df_1.fillna(0)

def break_df ( df_1 ):
    humor = df_1[['is_humor']]
    humor_rating = df_1[['humor_rating']]
    humor_contro = df_1[['humor_controversy']]
    offense_rating = df_1[['offense_rating']]
    return humor,humor_rating,humor_contro,offense_rating

def train_test ( X,y,z ):
    X_train, X_test, y_train, y_test = train_test_split(X, y , train_size = z)
    return X_train, X_test, y_train, y_test