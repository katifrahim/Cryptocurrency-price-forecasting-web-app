from django.shortcuts import render, redirect
from .forms import SignupForm, LoginForm, EditProfileForm, PasswordChange, AiModelForm
from django.contrib.auth import login, authenticate, logout, update_session_auth_hash
from django.contrib.auth.models import User
from .models import User_Logs
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from datetime import datetime, date
import requests
from newspaper import Article
from textblob import TextBlob
import nltk
import json
import base64
from io import BytesIO
import seaborn as sns
matplotlib.use('Agg')

def home_view(request):
    # The template is rendered when the user trys to access the page via url which sends request with the GET method to this view function
    return render(request, 'home.html')

def docs_view(request):
    # The template is rendered when the user trys to access the page via url which sends request with the GET method to this view function
    return render(request, 'docs.html')

def signup_view(request):
    if request.method == 'POST': # This code block will run once the user fills and submits the form present in the template which sends request with the POST method to this view function
        form = SignupForm(request.POST) # Creates an instance of the form and fills its fields with the data received from the request
        if form.is_valid(): # This code block runs if the form fulfills the validation criteria stated in forms.py for this form
            form.save() # Runs the 'save' method stated in the forms.py for this form
            return redirect('/login/') # The user is redirected to the login page
    else: # This code block runs when the user trys to access the page via url which sends request with the GET method to this view function
        form = SignupForm() # Creates an instance of the empty form 
    # The template is rendered and the default form is sent to it via context to be displayed
    return render(request, 'signup.html', {'form':form})

def login_view(request):
    if request.method == 'POST': # This code block will run once the user fills and submits the form present in the template which sends request with the POST method to this view function
        form = LoginForm(request.POST) # Creates an instance of the form and fills its fields with the data received from the request
        if form.is_valid(): # This code block runs if the form fulfills the validation criteria stated in forms.py for this form
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(request, username=username, password=password) # Verifying the user's username and pasword with the ones stored Django authentication system
            if user is not None: # This code block runs if the user is authenticated
                login(request, user) # Logging the user into the current session if authentication is successful
                return redirect('/model/') # The user is redirected to the AI model page
    else: # This code block runs when the user trys to access the page via url which sends request with the GET method to this view function
        form = LoginForm() # Creates an instance of the empty form 
    # The template is rendered and the default form is sent to it via context to be displayed
    return render(request, 'login.html', {'form':form})

def logout_view(request):
    if request.method == 'POST': # This code block will run once the user presses the logout button present in the profile page which sends a request with the POST method to this view function
        logout(request) # The user is logged out from the currennt session
        return redirect('/') # The user is redirected to the home page
    else: # This code block runs when the user trys to access the page via url which sends request with the GET method to this view function
        return redirect('/') # The user is redirected to the home page

def profile_view(request):
    current_user_id = request.user.id # User id of the current user is obtained from the user's record(object)
    user_logs = User_Logs.objects.filter(user_id=current_user_id).order_by("-date") # The ORM is used to obtain and store user's logs and orders it in descending order according to the date of the logs
    # The template is rendered and the default user_logs data is sent to it via context to be displayed
    return render(request, 'profile.html', {'user_logs':user_logs})

def edit_profile_view(request):
    current_user = request.user
    if request.method == 'POST': # This code block will run once the user fills and submits the form present in the template which sends request with the POST method to this view function
        form = EditProfileForm(request.POST, instance=current_user) # Creates an instance of the form that is filled with user's previous information and updates it with the new information recieved from the request
        if form.is_valid(): # This code block runs if the form fulfills the validation criteria stated in forms.py for this form
            form.save() # Runs the 'save' method stated in the forms.py for this form
            return redirect('/profile/') # The user is redirected to the profile page
    else: # This code block runs when the user trys to access the page via url which sends request with the GET method to this view function
        form = EditProfileForm(instance=current_user) # Creates an instance of the form that is filled with user's previous information so that it can be edited
    # The template is rendered and the default form is sent to it via context to be displayed
    return render(request, 'edit_profile.html', {'form':form})

def password_change_view(request):
    if request.method == 'POST': # This code block will run once the user fills and submits the form present in the template which sends request with the POST method to this view function
        form = PasswordChange(request.user, request.POST) # Creates an instance of the form, fills its fields with the data received from the request and sends the current user's record(object) to the form
        if form.is_valid(): # This code block runs if the form fulfills the validation criteria stated in forms.py for this form
            user_record = form.save() # Runs the 'save' method stated in the forms.py for this form
            update_session_auth_hash(request, user_record) # It is used to update user's session authentication hash, otherwise the user will automatically be logged out and will be required to relogin to update the session authentication hash
            return redirect('/profile/') # The user is redirected to the profile page
    else: # This code block runs when the user trys to access the page via url which sends request with the GET method to this view function
        form = PasswordChange(request.user) # Creates an instance of the form and sends the current user's record(object) to the form
    # The template is rendered and the default form is sent to it via context to be displayed
    return render(request, 'password_change.html', {'form':form})
    

def ai_model_view(request):
    
    if request.method == 'POST': # This code block will run once the user fills and submits the form present in the template which sends request with the POST method to this view function
        form = AiModelForm(request.POST) # Creates an instance of the form and fills it with the data received from the request
        if form.is_valid(): # This code block runs if the form fulfills the validation criteria stated in forms.py for this form
            # Storing the user inputs in respective variables
            selected_crypto = form.cleaned_data['cryptocurrency']
            selected_day = form.cleaned_data['n_ahead']
            n_ahead = int(selected_day)

            crypto_accuracy_data = {
                'BTC-USD':{'selected_crypto_name':'bitcoin', 'Overall_Accuracy':'48.896632%', 'Up_Trend_Accuracy':'48.936170%', 'Down_Trend_Accuracy':'48.858447%'},
                'ETC-USD':{'selected_crypto_name':'ethereum', 'Overall_Accuracy':'49.739130%', 'Up_Trend_Accuracy':'49.668874%', 'Down_Trend_Accuracy':'49.816850%'},
                'DOGE-USD':{'selected_crypto_name':'dogecoin', 'Overall_Accuracy':'50.174216%', 'Up_Trend_Accuracy':'52.611940%', 'Down_Trend_Accuracy':'48.039216%'}
            }
            
            # Performing linear search algorithm to obtain accuracy data of the selected cryptocurrency from the above dictionary
            crypto_data_found = False
            for crypto_code, accuracy_data in crypto_accuracy_data.items():
                if crypto_code == selected_crypto:
                    selected_crypto_name = accuracy_data['selected_crypto_name']
                    Overall_Accuracy = accuracy_data['Overall_Accuracy']
                    Up_Trend_Accuracy = accuracy_data['Up_Trend_Accuracy']
                    Down_Trend_Accuracy = accuracy_data['Down_Trend_Accuracy']
                    crypto_data_found = True

            if not crypto_data_found:
                selected_crypto_name = 'Error'
                Overall_Accuracy = 'Error'
                Up_Trend_Accuracy = 'Error'
                Down_Trend_Accuracy = 'Error'

            # Downloading the histocial Open-High-Low-Close(OHLC) data of the selected cryptocurrency from Yahoo Finance API and storing it in a pandas dataframe
            df = yf.download(selected_crypto)

            # Using pagination on the Guardian API to obtain articles from the guardian news that are related to the selected cryptocurrency 
            starting_date = df.index[0].strftime('%Y-%m-%d') # Stores the first date of historical data present in the dataframe and formats it according to the Y-M-D format.
            # Defining a query parameter for the API request
            params = {'api-key':'aa10117e-90e9-4a25-845f-4b5d584958f7',
                       'q':selected_crypto_name, # Articles containing the selected cryptocurrency will be filtered
                       'order-by':'newest', 
                       'from-date':starting_date, # The date-range starts from first date of the historical data present in the dataframe
                       'to-date':datetime.now().strftime('%Y-%m-%d'), # The date-range ends at the current date
                       'page-size':200} # 200 numbers of articles will be returned in each response page (Which is the max limit of this API)
            # Pagination
            all_articles = []
            current_page = 1
            while True:
                # Update the page parameter in the request
                params['page'] = current_page # Defining the page number of the response in the query parameter

                result = requests.get('https://content.guardianapis.com/search', params=params) # Sending request to the API with the query parameter using GET method and getting response from the API which is stored in a variable
                result = result.json()
                all_articles.extend(result['response']['results']) # Add the articles to the overall list
                if current_page * params['page-size'] < result['response']['total']: # If the number of articles retrieved are less than the total number of articles present in the response, the current page number is incremented by 1 to shift to the next page in the next iteration
                    current_page += 1
                else:
                    break  # Exit the loop if all articles have been retrieved
            
            # Retrieving and storing the urls and publication-dates of the articles because only the urls and the publication-dates of the articles are needed to begin with the sentiment analysis
            urls = []
            dates = []
            for i in all_articles:
                urls.append(i["webUrl"])
                dates.append(pd.Timestamp(pd.to_datetime(i['webPublicationDate']).date()))

            # Performing sentiment analysis on all the articles obtained from the Guardian API
            nltk.download('punkt') # Punkt is a data package within Natural-Language-ToolKit(NLTK) library that is used for tokenizing (breaking down text into individual word)
            df['Polarity'] = 0 # Storing the empty cells of this column with 0 because 0 represents neutral sentiment
            for i in range(len(urls)): # This loop iterates for each article that was retrieved using Guardian API
                if dates[i] in df.index: # This code block runs if the publication-date of the article falls within the date-range of the dataframe
                    # Obtaining the summary of the articles
                    url = urls[i]
                    article = Article(url) # Creates an article instance of the given url
                    article.download() # Downloads the article
                    if article.download_state != 2:  # Skip to the next iteration if the article fails to download (2 represents successful download)
                        continue
                    article.parse() # Breaking the text into its components e.g. title, authors, images, etc
                    article.nlp() # Applies Natural Language Processing to the article to work with the human language data.
                    summary = article.summary

                    # Performing the sentiment analysis on the summary of the article
                    blob = TextBlob(summary) # Creates a textblob instance of the summary
                    sentiment = blob.sentiment.polarity # Identifies and stores the sentiment expressed in the summary of the article from a range of -1.0 to 1.0. 
                    df.loc[dates[i], 'Polarity'] = sentiment # Setting the sentiment polarity value in the 'Polarity' column of the dataframe at the publication-date of the article

            # Pre-processing of the data
            df['Polarity'].fillna(0, inplace=True) # Fills the Not a Number(NaN) values present in the 'Polarity' column with 0
            df = df.drop(['Adj Close','Volume'], axis='columns') # Removing the redundant columns from the dataframe
            timeseries_values = df.values.astype('float32') # Converting the dataframe to a numpy array
            scaler = StandardScaler()
            timeseries = scaler.fit_transform(timeseries_values) # Scales the values of the numpy array so that they have a mean of 0 and and standard deviation of 1. If mean is around 0 then it is easier for the model to learn values and prevents bias and the standard deviation is used to compare the scale of different values.
            feature_amount = timeseries[0].size # It basically stores the number of columns that were present in the pandas dataframe

            # Creating in/out sequence data
            inout_seq = []
            window_size = 50
            L = len(timeseries) # Stores total number of rows present in the data
            for i in range(L-window_size-1): # i represents a day
                train_seq = timeseries[i:i+window_size] # This stores the features which is basically the data of the days of window size(e.g. 50 days)
                train_label = timeseries[i+window_size:i+window_size+1] # This stores the targets which is basically the data of the next day(e.g.51th day)
                inout_seq.append((train_seq , train_label)) # The features and the labels for each of these iterations are appended to the list

            # Convert inout_seq to a tensor dataset inside the list comprehension
            dataset = data.TensorDataset(
                torch.stack([torch.FloatTensor(s[0]) for s in inout_seq]),
                torch.stack([torch.FloatTensor(s[1]) for s in inout_seq]))

            # Creating a dataloader which allows us to retrieve data items from the dataset for training the AI model
            train_loader = data.DataLoader(dataset, shuffle=True, batch_size=16, drop_last=True) # This dataloader will randomly select 16 data items in each batch from the tensor dataset when it is used in the training process of the model

            # Creating the LSTM model
            class PricePredictorLSTM(nn.Module): # Defining an deep learning model using the neural network(nn) library of pytorch
                def __init__(self, input_size, output_size): # Initializes and defines the parameters and the layers of this model 
                    super(PricePredictorLSTM, self).__init__()
                    # The size of the layer are approximated using the structure of a typical neural networks 
                    self.input_size = input_size
                    self.hidden_layer_1_size = input_size*25
                    self.hidden_layer_2_size = input_size*10
                    self.hidden_layer_3_size = input_size*5
                    self.output_size = output_size
                    self.lstm = nn.LSTM(self.input_size, self.hidden_layer_1_size, batch_first=True) # Defining the Long Short-Term Memory(LSTM) layer of the model with respective to its input_size and output_size values
                    self.relu = nn.ReLU() # Defining the Rectified Linear Unit(ReLU) activation function of the model
                    # Defining 3 Linear layer of the model with respective to its input_size and output_size values
                    self.linear = nn.Linear(self.hidden_layer_1_size, self.hidden_layer_2_size)
                    self.linear2 = nn.Linear(self.hidden_layer_2_size, self.hidden_layer_3_size)
                    self.linear3 = nn.Linear(self.hidden_layer_3_size, self.output_size)

                def forward(self, input_seq): # Defines how the data flows through the layers of the LSTM model
                    out, _ = self.lstm(input_seq) # The output from the LSTM layer for each timestep in the input_seq is stored in 'out' tensor, while the hidden state and the cell state of the LSTM is stored in the '_' variables
                    out = self.relu(out) # Applying the activation function the output of the LSTM layer
                    out = self.linear(out[:,-1,:]) # The last timestep of the LSTM output is used in the 1st Linear layer
                    out = self.relu(out) # Applying the activation function the output of the Linear layer
                    out = self.linear2(out) # The output passes through the 2nd Linear layer
                    out = self.relu(out) # Applying the activation function the output of the Linear2 layer
                    out = self.linear3(out) # The output passes through the 3rd Linear layer
                    return out 

            # Instanciating the model
            model = PricePredictorLSTM(input_size=feature_amount,output_size=feature_amount)

            # Training the model
            loss_function = nn.MSELoss() # This is used to measure the loss of the predictions made by the model
            optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4) # This is used to adjust the model's parameters to minimise the loss associated with the predictions that is measured by the loss function
            epochs = 300 # The model will be trained 300 times
            for i in range(epochs): # This for loop runs 300 times
                for seq, labels in train_loader: # This loop runs for each batch of dataset present in the dataloader
                    seq, labels = seq, labels.view(-1, feature_amount) # Here,the features and the reshaped targets of the data items of the batch are retrieved and stored in their respective variables
                    model.train() # Setting the model in training mode
                    optimizer.zero_grad() # Clears the gradients of all parameters of the model to prevent gradient from accumulating
                    y_pred = model(seq) # The model makes predictions on the features of the data items
                    loss = loss_function(y_pred, labels) # The predicted and the actual target values are compared in the loss function to calculate the Mean Squard Error Loss(MSELoss) of the predictions
                    loss.backward() # Calculating gradients of the loss according the model's parameters
                    optimizer.step() # Gradient descent: Optimizing model's parameters according to the gradients of the loss.
                # Reducing the learning rate and increase the regularization as time goes on
                if i == 100:
                   for g in optimizer.param_groups:
                        g['lr'] = 0.005
                elif i == 200:
                    for g in optimizer.param_groups:
                        g['lr'] = 0.0001
                        g['weight_decay'] = 1e-4
    
            # Using the trained model to make predictions
            def extrapolate_n_days_ahead(sequence, n): # This predicts the cryptocurrency's price till n days ahead
                model.eval() # Setting the model in evaluation mode
                with torch.no_grad(): # Turning off gradients for validation, saves memory and computation
                    for i in range(n): # This for loop runs for each day of the n_ahead days to predict the cryptocurrency's price for each day untill it reaches the nth day ahead.
                        preds = model(sequence) # Outputs the predicted price of the next day
                        sequence = torch.cat((sequence, preds.unsqueeze(1)), dim=1) # The predicted cryptocurrency's price of the next day is added into the sequence data to be used as features in the next iteration to predict the cryptocurrency's price of the other days untill it reaches the nth day ahead.
                    return sequence

            # Post-processing of the data
            metric_to_look_at = 3 # 0 for Open, 1 for High, 2 for Low, 3 for Close(target)
            features_seq = torch.stack([torch.FloatTensor(s[0]) for s in inout_seq]) # This code retrieves the tensor dataset containing only the features 
            data_items = features_seq[:1] # This code retrieves the most recent data item(window_size = 50 days) from the features sequence
            data_items_array = np.array(data_items.cpu()).squeeze()[:,metric_to_look_at] # Here the data items are converted to a numpy array, reshaped and then the closing price is selected from it to be plotted in the graph
            extrapolated_array = extrapolate_n_days_ahead(data_items, n_ahead).squeeze(0)[:,metric_to_look_at].cpu() # The predicted prices are reshaped and then the closing price is selected from it to be plotted in the graph

            # Identifying the trend of the predicted price
            last_item_value = abs(data_items_array[-1]) # The most recent actual closing price
            last_extrapolated_value = abs(extrapolated_array[-1].numpy()) # The predicted closing price at the nth day ahead
            # The rational for placing the signs of inequality inversely below is because the values are still scaled and not transformed back to its orginal form.
            if(last_item_value > last_extrapolated_value):
                signal = 'Up'
            elif(last_item_value < last_extrapolated_value):
                signal = 'Down'
            else:
                signal = 'Flat'
            Trend = f'{signal} Trend in the upcoming {selected_day} Days'

            # Creating the predictiion graph using the matplotlib library
            sns.set_style('whitegrid')
            plt.figure(figsize=(15,6))
            plt.plot(extrapolated_array, label='Predicted Close Price')
            plt.plot(data_items_array, label='Actual Close Price')
            plt.ylabel('Scaled Price')
            plt.xlabel('Days')
            plt.legend()

            # Encoding the prediction graph image to a base64 text and sending it to the template
            buffer = BytesIO() # This allows to temporarily store data in memory 
            plt.savefig(buffer, format='png') # Uses the buffer defined above to temporarily store the plotted graph image in PNG format
            buffer.seek(0) # Temporarily saves the data stored in the buffer to the memory
            image_png = buffer.getvalue() # Retrieves the image of the graph from the temporary memory and stores it in a variable
            image_base64 = base64.b64encode(image_png) # Encodes the image as base64 bytes object
            image_base64 = image_base64.decode('utf-8') # The base64 encoded bytes object is decoded to string text format to send it to the template
            buffer.close() # The data stored and saved temporarily in the memory is freed

            # Creating a dictionary that stores information of the results of the prediction
            results={
                'Graph': image_base64,
                'Trend': Trend,
                'Overall_Accuracy': Overall_Accuracy,
                'Up_Trend_Accuracy': Up_Trend_Accuracy,
                'Down_Trend_Accuracy': Down_Trend_Accuracy,
            }

            # Inserting a new record in the User Logs tabel which contains information regarding the results of the predictions
            current_user_id = request.user.id # The user id is used as a foreign key in the database table that the 'User_Logs' model represents
            User_Logs_Instance = User_Logs(
                user_id = User.objects.get(id=current_user_id), # Using Django's ORM and user id to retrieve the record(object) of the current user from the database table that the 'User' model represents
                input_coin = selected_crypto,
                input_days = selected_day,
                output_trend = f'{signal} Trend',
                date = date.today()
            ) # An instance of the 'User_Logs' model is created which represents a record in the database table of the model
            User_Logs_Instance.save() # Saving the instance of the model which inserts and saves the record into the database table that the model represents
            
            # The AI model template is rendered and the prediction results and the form is sent to the template via context to be displayed
            return render(request, 'model.html', {'results': results, 'form': form})
        
    else: # This code block runs when the user trys to access the page via url which sends request with the GET method to this view function
        form = AiModelForm() # Creates an instance of the empty form 
    # The template is rendered and the default form is sent to it via context to be displayed
    return render(request, 'model.html',{'form':form})