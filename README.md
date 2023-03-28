# Microbusiness Density Forecast
The goal of this project is to predict monthly microbusiness density in a given area. Developing an accurate model trained on U.S. county-level data.

## Dataset
Dataset can be obtained from kaggle competition. In this repository I use kaggle API to pull the dataset `kaggle competitions download -c godaddy-microbusiness-density-forecasting`

## Notebook
The notebook.ipynb file contains the code used to preprocess the data, build the model, train it, and generate forecasts. The notebook is divided into the following sections:
- Data Loading and Exploration
- Data Preprocessing
- Model Architecture
- Model Training
- Model Evaluation and Forecasting

## Model Architecture
The model used in this project is a deep learning model based on Long Short-Term Memory (LSTM) neural networks. The model consists of 6 LSTM layers, followed by a dense layer that outputs the forecast. The model is trained using the mean absolute error (MAE) loss function and the Adam optimizer.

## Future Work
In this project, my main focus was on building the machine learning model and gaining a better understanding of the overall flow of an ML project. As a result, there is limited information regarding why I chose a specific architecture for the model or the feature selection process. Moving forward, I plan to dedicate more time to improving the accuracy of the model and conducting a more in-depth analysis of both the dataset and the various model selection strategies available.
