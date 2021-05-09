import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

from django.shortcuts import render
from .utils import get_plot

# print("All libraries imported") 
# Create your views here.
def home(request): 
    return render(request, 'home.html')

def predict(request): 
    stock_name = request.POST['stock_name']
    prediction_day = request.POST['prediction_day']
    # ML algorithm will go here!
    df = pd.read_csv(r"C:\Users\user\Desktop\DS\Projects\django_projects\stock_price_prediction\predictor\TCS.csv",encoding='utf-8')
    df = df.dropna()
    train_data, test_data = df[0:int(len(df)*0.7)], df[int(len(df)*0.7):]
    training_data = train_data['Close'].values
    test_data = test_data['Close'].values
    history = [x for x in training_data]
    model_predictions = []
    N_test_observations = len(test_data)
    for time_point in range(N_test_observations):
        model = ARIMA(history, order=(4,1,0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        model_predictions.append(yhat)
        true_test_value = test_data[time_point]
        history.append(true_test_value)
    MSE_error = mean_squared_error(test_data, model_predictions)
    # print('Testing Mean Squared Error is {}'.format(MSE_error))
    test_set_range = df[int(len(df)*0.7):].index
    #plt.plot(test_set_range, model_predictions, color='blue', marker='o', linestyle='dashed',label='Predicted Price')
    #plt.plot(test_set_range, test_data, color='red', label='Actual Price')
    chart1 = get_plot(test_set_range, model_predictions, 'Predicted Price')
    chart2 = get_plot(test_set_range, test_data, 'Actual Price')
    args = {}
    args['stock_name'] = stock_name
    args['mean_squared_error'] = MSE_error 
    args['chart1'] = chart1 
    args['chart2'] = chart2
    return render(request, 'output.html', args)