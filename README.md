# Cryptocurrency Price Forecasting Web App
## Overview
This web app is built using Django. It has a built-in python-based Long Short-Term Memory (LSTM) model that uses the entire historical Open-High-Low-Close (OHLC) and news-sentiment-analysis data of the choosen cryptocurrency to automatically perform technical and fundamental analysis on it and forecast its future price till upto 15 days ahead with over 50% accuracy.

![Screenshot 2025-06-24 191659](https://github.com/user-attachments/assets/0f1ba3a7-3ef7-4ad0-879d-19bed4789c84)

## Features
- Real-time price forecasting of bitcoin, ethereum and dogecoin till 5, 10 and 15 days ahead with over 50% accuracy.
- Dashboard with the chart, accuracy and trend of the forecasts.
- User authentication system.
- User history.
- User docs.

## Tech Stack
- Python
- Django 
- Pytorch (for LSTM)
- Sci-kit learn (for data preprocessing)
- Matplotlib (for graphs)
- Natural Language ToolKit, Textblob and Newspaper3K (for sentiment analysis)
- Yahoo Finance API (for fetching OHLC data)
- Guardian News API (for fetching news data)
- SQLite Database with Object Relational Mapping (ORM)

## Design
### User Interface (UI):
[Web App UI Design Overview in Figma](https://www.figma.com/design/QBmbxszGzhhZtziTtdYsho/Web-App-UI-Design-Overview?node-id=0-1&t=MFBIG7MIZlCv3lXU-1)
### System Flowchart:
![System Flowchart](https://github.com/user-attachments/assets/b0b80f8e-5e7d-42e2-b9b4-89dbce178b81)
### Entity Relationship Diagram:
![image](https://github.com/user-attachments/assets/cb7e8754-cdd4-4ce6-8804-3dd0196d9e9e)

## Setup instructions
1. Open the terminal/ command prompt
2. Check the python version of your system using by typing `python --version`
3. If python is not installed on your system or if it's not up-to-date, download and install it from [python.org](https://www.python.org/)
4. During the installation, check the box for *Add Python to PATH* and choose the option to install the *Python Launcher*
5. Download venv by typing `pip install venv` in terminal/command prompt
6. Navigate to the WebApp folder using `cd`. For example: `cd WebApp`
7. Create the venv by typing `python -m venv venv`
8. Then, to activate that venv in WINDOWS type `WebApp\cripts\activate.bat` within the same directory
10. To activate that venv in MacOS type `source venv/bin/activate` within the same directory
11. To install the dependencies of the web application, type `pip install -r requirements.txt`
12. Type the following command to initialize the web application : `python manage.py migrate`
13. Type the following command to run the Django development server : `python manage.py runserver`
14. Open the web browser and go to [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
