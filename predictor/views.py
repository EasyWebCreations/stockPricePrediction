from django.shortcuts import render

# Create your views here.
def home(request): 
    return render(request, 'home.html')

def predict(request): 
    stock_name = request.POST['stock_name']
    prediction_day = request.POST['prediction_day']
    print("stock_nam and prediction time is: ", stock_name, prediction_day)
    # ML algorithm will go here!
    args = {}
    prediction_result = int(prediction_day)*10
    args['stock_name'] = stock_name
    args['prediction_day'] = prediction_day 
    args['to_show'] = prediction_result
    return render(request, 'output.html', args)