"""
This class contains the functions to create, run, and test various ML
models to try to predict used car prices.

By Cole Koryto
"""

import pprint
import pandas as pd
from pathlib import Path
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from joblib import dump, load
from tensorflow.keras.models import load_model
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer


class LearningModels:

    # creates instance variables for k-neighbors model
    def __init__(self, car_df):

        # gets total dataset
        x = car_df.drop(columns=["Price"])
        y = car_df.loc[:, "Price"]

        # splits data into training and test set
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size = 0.10, random_state=17, shuffle=True)

        # scales data with standard scaler
        self.scaleData()


    # scales data with standard scaler
    def scaleData(self):

        # loads existing scaler if possible, otherwise creates
        print("\nScaling data")
        scaler_file = Path("TrainingScaler.joblib")
        if scaler_file.exists():
            std_scaler = load("TrainingScaler.joblib")
        else:
            std_scaler = StandardScaler()
            std_scaler.fit(self.x_train)

        columns = self.x_train.columns
        self.x_train = std_scaler.transform(self.x_train)
        self.x_test = std_scaler.transform(self.x_test)
        self.x_train = pd.DataFrame(self.x_train, columns=columns)
        self.x_test = pd.DataFrame(self.x_test, columns=columns)

        # saves scaler
        dump(std_scaler, 'TrainingScaler.joblib')

    # outputs metrics for given predictions and actual data set
    def outputMetrics(self, y_actual, y_pred):
        mae = metrics.mean_absolute_error(y_actual, y_pred)
        mse = metrics.mean_squared_error(y_actual, y_pred)
        rmse = metrics.mean_squared_error(y_actual, y_pred, squared=False)
        r2 = metrics.r2_score(y_actual, y_pred)
        print("--------------------------------------")
        print('MAE is {}'.format(mae))
        print('MSE is {}'.format(mse))
        print('RMSE is {}'.format(rmse))
        print('R2 score is {}'.format(r2))
        print("--------------------------------------")

    # creates, tests, and visualizes a k-neighbors regression
    def createKNeighborsModel(self):

        # loops through a range of k to find the best model
        print("\n\nCreating k-neighbors regression model")
        parameters = [{'n_neighbors': [1, 2, 3, 4, 5, 6], 'n_jobs': [-1]}]
        K = 5
        k_neighbors_reg = GridSearchCV(KNeighborsRegressor(), parameters, cv=K, verbose=4, n_jobs=-1)
        k_neighbors_reg.fit(self.x_train, self.y_train)
        print(f"Best model parameters: {k_neighbors_reg.best_params_}")
        print("\nTraining Set Metrics")
        y_train_pred = k_neighbors_reg.predict(self.x_train)
        self.outputMetrics(self.y_train, y_train_pred)
        print("\nTest Set Metrics")
        y_test_pred = k_neighbors_reg.predict(self.x_test)
        self.outputMetrics(self.y_test, y_test_pred)

        # plots a graph comparing actual value versus predicted value
        fig, ax = plt.subplots()
        y_pred = k_neighbors_reg.predict(self.x_test)
        ax.scatter(y_pred, self.y_test, edgecolors=(0, 0, 1), alpha=0.1)
        ax.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=3)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        plt.show()

        # saves model
        dump(k_neighbors_reg, 'KNN.joblib')


        # creates, tests, and visualizes a linear regression (elastic net regression)
    def createLinearModel(self):

        # creates a linear regression
        print("\nCreating linear regression model (elastic net)")
        parameters = [{'l1_ratio': [0.9, 0.95, 1]}]
        K = 5
        linearRegElastic = GridSearchCV(ElasticNet(random_state=17), parameters, cv=K, verbose=4, n_jobs=-1)
        linearRegElastic.fit(self.x_train, self.y_train)
        print(f"Best model parameters: {linearRegElastic.best_params_}")
        print("\nTraining Set Metrics")
        y_train_pred = linearRegElastic.predict(self.x_train)
        self.outputMetrics(self.y_train, y_train_pred)
        print("\nTest Set Metrics")
        y_test_pred = linearRegElastic.predict(self.x_test)
        self.outputMetrics(self.y_test, y_test_pred)

        # plots a graph comparing actual value versus predicted value
        fig, ax = plt.subplots()
        y_pred = linearRegElastic.predict(self.x_test)
        ax.scatter(y_pred, self.y_test, edgecolors=(0, 0, 1), alpha=0.1)
        ax.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=3)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        plt.show()

        # saves model
        dump(linearRegElastic, 'Linear.joblib')


    # creates, tests, and visualizes a SVM regression
    def createSVMModel(self):

        # creates a SVM polynomial model
        print("\nCreating SVM regression model")
        parameters = [{'C': [0.1, 1, 10, 100], 'degree': [2, 3, 4, 5], 'cache_size': [3000], 'verbose': [True]}]
        K = 5
        svm_poly_reg = GridSearchCV(SVR(), parameters, cv=K, verbose=4, n_jobs=-1)
        svm_poly_reg.fit(self.x_train, self.y_train.ravel())
        print(f"Best model parameters: {svm_poly_reg.best_params_}")
        print("\nTraining Set Metrics")
        y_train_pred = svm_poly_reg.predict(self.x_train)
        self.outputMetrics(self.y_train, y_train_pred)
        print("\nTest Set Metrics")
        y_test_pred = svm_poly_reg.predict(self.x_test)
        self.outputMetrics(self.y_test, y_test_pred)

        # plots a graph comparing actual value versus predicted value
        fig, ax = plt.subplots()
        y_pred = svm_poly_reg.predict(self.x_test)
        ax.scatter(y_pred, self.y_test, edgecolors=(0, 0, 1), alpha=0.1)
        ax.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=3,)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        plt.show()

        # saves model
        dump(svm_poly_reg, 'SVM.joblib')

    # creates, tests, and visualizes a XGBoost regression
    def createXGBoostModel(self):
        # creates a SVM polynomial model
        print("\nCreating XGBoost regression model")
        parameters = [{}]
        K = 5
        xgb_reg = GridSearchCV(XGBRegressor(), parameters, cv=K, verbose=4, n_jobs=-1)
        xgb_reg.fit(self.x_train, self.y_train.ravel())
        print(f"Best model parameters: {xgb_reg.best_params_}")
        print("\nTraining Set Metrics")
        y_train_pred = xgb_reg.predict(self.x_train)
        self.outputMetrics(self.y_train, y_train_pred)
        print("\nTest Set Metrics")
        y_test_pred = xgb_reg.predict(self.x_test)
        self.outputMetrics(self.y_test, y_test_pred)

        # plots a graph comparing actual value versus predicted value
        fig, ax = plt.subplots()
        y_pred = xgb_reg.predict(self.x_test)
        ax.scatter(y_pred, self.y_test, edgecolors=(0, 0, 1), alpha=0.1)
        ax.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=3, )
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        plt.show()

        # plot feature importance
        sorted_idx = xgb_reg.best_estimator_.feature_importances_.argsort()
        sorted_df = pd.DataFrame(sorted_idx, columns=self.x_train.columns)
        plt.barh(sorted_df)
        plt.xlabel("Xgboost Feature Importance")
        plot_importance(xgb_reg.best_estimator_)
        plt.show()

        # saves model
        dump(xgb_reg, 'XGBR.joblib')


    # creates, tests, and visualizes a sequential ANN
    def createNeuralNetwork(self):

        # builds neural network
        tf.random.set_seed(17)
        neuralModel = tf.keras.Sequential()
        neuralModel.add(tf.keras.layers.InputLayer(input_shape=[788]))
        neuralModel.add(tf.keras.layers.Dense(1000, activation="relu"))
        neuralModel.add(tf.keras.layers.Dense(1000, activation="relu"))
        neuralModel.add(tf.keras.layers.Dense(1000, activation="relu"))
        neuralModel.add(tf.keras.layers.Dense(1000, activation="relu"))
        neuralModel.add(tf.keras.layers.Dense(1000, activation="relu"))
        neuralModel.add(tf.keras.layers.Dense(1, kernel_initializer='normal'))

        # compiles neural network
        neuralModel.compile(loss="mean_squared_error", optimizer="adam")

        # trains neural network
        history = neuralModel.fit(self.x_train, self.y_train, epochs=300)

        # evaluates neural network
        print(f"Loss and accuracy for test set: {neuralModel.evaluate(self.x_test, self.y_test)}")
        print("\nTraining Set Metrics")
        y_train_pred = neuralModel.predict(self.x_train)
        self.outputMetrics(self.y_train, y_train_pred)
        print("\nTest Set Metrics")
        y_test_pred = neuralModel.predict(self.x_test)
        self.outputMetrics(self.y_test, y_test_pred)

        # plots a graph comparing actual value versus predicted value
        fig, ax = plt.subplots()
        y_pred = neuralModel.predict(self.x_test)
        ax.scatter(y_pred, self.y_test, edgecolors=(0, 0, 1), alpha=0.1)
        ax.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=3, )
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        plt.show()

        # outputs the learning curves
        pd.DataFrame(history.history).plot(
            figsize=(8, 5), xlim=[0, 300], ylim=[0, 900000], grid=True, xlabel="Epoch",
            style=["r--", "r--.", "b-", "b-*"])
        plt.legend(loc="lower left")  # extra code
        plt.show()

        # saves model
        neuralModel.save("neural_model.h5")

    # run all models and output predictions
    def runModels(self):

        # imports user dataset
        print("Getting and cleaning user data set")
        input_df = pd.read_csv("Model_Inputs.csv")

        # cleans dataset and removes unwanted features
        input_df = input_df.dropna()

        # one hot encodes categorical features
        print("Encoding categorical features")
        transformer = make_column_transformer(
            (OneHotEncoder(sparse_output=False), ["Maker", "Genmodel", "Color", "Bodytype", "Gearbox", "Fuel_type"]),
            verbose_feature_names_out=False,
            remainder='passthrough')
        transformed_array = transformer.fit_transform(input_df)
        input_df = pd.DataFrame(transformed_array, columns=transformer.get_feature_names_out())

        # loads all models
        print("Loading models")
        k_neighbors_reg = load('KNeighbors.joblib')
        linearRegElastic = load('Linear.joblib')
        svmRegElastic = load('SVM.joblib')
        xgboost_reg = load('XGBR.joblib')
        neuralModel = load_model('neural_model.h5')

        # loads scaler
        std_scaler = load("TrainingScaler.joblib")

        # predicts with each model and outputs prediction
        for index, x_new in input_df.iterrows():
            x_new_scaled = std_scaler.transform([x_new])
            print(x_new_scaled)
            print(f"K Neighbors price prediction: {k_neighbors_reg.predict(x_new_scaled)}")
            print(f"Linear model price prediction: {linearRegElastic.predict(x_new_scaled)}")
            print(f"SVM model price prediction: {svmRegElastic.predict(x_new_scaled)}")
            print(f"XGBoost price prediction: {xgboost_reg.predict(x_new_scaled)}")
            print(f"Neural network rent prediction: {neuralModel.predict(x_new)}")
