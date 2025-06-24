# Cryptocurrency Price Forecasting Web App
## Overview
This web app is built using Django. It has a built-in Long Short-Term Momeory (LSTM) model that uses the Open-High-Low-Close (OHLC) data and News sentiment analysis data of the choosen cryptocurrency to automatically perform technical and fundamental analysis on it and predict its future price till upto 15 days into the future with above 50% accuracy.

![Screenshot 2025-06-24 191659](https://github.com/user-attachments/assets/0f1ba3a7-3ef7-4ad0-879d-19bed4789c84)

## Key Tech Stack
- Python
- Django 
- Pytorch (for LSTM)
- Sci-kit learn (for data preprocessing)
- Matplotlib (for graphs)
- Natural Language ToolKit, Textblob and Newspaper3K (for sentiment analysis)
- Yahoo Finance API (for fetching OHLC data)
- Guardian News API (for fetching news data)
- SQLite Database

## Design and Plan
### User Interface (UI):
[Web App UI Design Overview in Figma](https://www.figma.com/design/QBmbxszGzhhZtziTtdYsho/Web-App-UI-Design-Overview?node-id=0-1&t=MFBIG7MIZlCv3lXU-1)
### System Flowchart:
![System Flowchart](https://github.com/user-attachments/assets/b0b80f8e-5e7d-42e2-b9b4-89dbce178b81)
### Entity Relationship Diagram:
![image](https://github.com/user-attachments/assets/cb7e8754-cdd4-4ce6-8804-3dd0196d9e9e)


