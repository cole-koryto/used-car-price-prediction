"""
Main code to call methods from the LearningModels class to
predict used car prices using ML models.

By Cole Koryto
"""

import pprint
import traceback
from LearningModels import *
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer


# gets and cleans dataset
def getDataset():

    # imports dataset
    print("Getting and cleaning data set")
    car_df = pd.read_csv("used_car_data.csv")

    # cleans dataset and removes unwanted features
    car_df = car_df.drop(columns=["Genmodel_ID", "Adv_ID", "Adv_year", "Adv_month"])
    car_df = car_df.dropna()
    car_df = car_df.drop(car_df[car_df["Annual_Tax"] == "*"].index)

    # strips unit labels from data
    car_df['Engin_size'] = car_df['Engin_size'].str.replace(r'[^\d.]', '', regex=True).astype(float)
    car_df['Average_mpg'] = car_df['Average_mpg'].str.replace(r'[^\d.]', '', regex=True).astype(float)
    car_df['Top_speed'] = car_df['Top_speed'].str.replace(r'[^\d.]', '', regex=True).astype(float)
    car_df['Runned_Miles'] = car_df['Runned_Miles'].str.replace(r'[^\d.]', '', regex=True).astype(float)
    car_df['Annual_Tax'] = car_df['Annual_Tax'].str.replace(r'[^\d.]', '', regex=True).astype(float)

    # one hot encodes categorical features
    print("Encoding categorical features")
    transformer = make_column_transformer(
        (OneHotEncoder(sparse_output=False), ["Maker", "Genmodel", "Color", "Bodytype", "Gearbox", "Fuel_type"]),
        verbose_feature_names_out=False,
        remainder='passthrough')
    transformed_array = transformer.fit_transform(car_df)
    car_df = pd.DataFrame(transformed_array, columns=transformer.get_feature_names_out())

    # returns final data set
    return car_df


# visualizes the data to understand structure of data set
def visualizeData(car_df):
    print(car_df.describe())
    print(car_df.info())

    # investigates dataset by plotting histograms
    car_df.hist(figsize=(12, 8))
    plt.show()

    # displays correlation matrix with price
    corr_matrix = car_df.iloc[:, 775:].corr()
    corr_matrix["Price"].sort_values(ascending=False).plot.bar()
    plt.tight_layout()
    plt.show()

def main():

    # sets up time tracking of program
    startTime = time.time()
    cpuStartTime = time.process_time()

    # gets car data set for models
    car_df = getDataset()

    # visualizes data
    visualizeData(car_df)

    # creates new model instance
    modelObject = LearningModels(car_df)

    # creates K neighbors regression model
    modelObject.createKNeighborsModel()

    # creates linear regression model
    modelObject.createLinearModel()

    # creates SVM regression model
    modelObject.createSVMModel()

    # creates XGBoost regression model
    modelObject.createXGBoostModel()

    # creates sequential neural network model
    modelObject.createNeuralNetwork()

    # runs models on user input
    modelObject.runModels()

    # outputs program time taken in total time and time CPU time
    elapsedTime = time.time() - startTime
    executionTime = time.process_time() - cpuStartTime
    print("\n\n***Program Performance***")
    print("Elapsed time: " + str(elapsedTime) + " seconds")
    print("CPU execution time: " + str(executionTime) + " seconds (without I/O or resource waiting time)")

# runs all main code
if __name__ == "__main__":
    main()
