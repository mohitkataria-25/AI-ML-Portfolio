

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess(data_path:str, test_size: float = 0.2, val_size: float=0.1, random_state: int = 42):

    #read and load the dataset
    used_cars = pd.read_csv(data_path)

    #Drop rows with missing values for target variable 
    used_cars = used_cars[used_cars['Price'].notna()].copy()

    #Create Input and Output
    x = used_cars.drop(columns=['Price'])
    y = used_cars['Price']

    x = pd.get_dummies(x, drop_first=True)
    #Derive training dataset
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=(test_size+val_size), random_state=random_state)

    #calculate test size split
    test_size_split = test_size/(test_size+val_size)

    #split temp data into test and validation datasets
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=test_size_split, random_state=random_state)

    #Scale Numeric features
    #num_col = x_train.select_dtypes(include=[np.number]).columns
    num_col = ['mileage_num', 'power_num', 'engine_num', 'Year', 'Kilometers_Driven',  'Seats', 'New_Price']
    num_col = [c for c in num_col if c in x_train.columns]

    scaler = StandardScaler()
    scaler.fit(x_train[num_col])

    x_train_scaled = x_train.copy()
    x_val_scaled = x_val.copy()
    x_test_scaled = x_test.copy()

    x_train_scaled[num_col] = scaler.transform(x_train[num_col])
    x_val_scaled[num_col] = scaler.transform(x_val[num_col])
    x_test_scaled[num_col] = scaler.transform(x_test[num_col])

    #extract numaber of features
    feature_names = list(x_train_scaled.columns)

    #return final output
    return (x_train_scaled, x_val_scaled, x_test_scaled, y_train, y_val, y_test, scaler, feature_names)

