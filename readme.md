# Predicting Stock Price Using Python - Machine learning

Introduction
--------
**Disclaimer:** The material  is purely for educational and should not be taken as professional investment advice. Invest at your own discretion.

Here in this Repo ,we have created an artificial neural network called Long Short Term Memory**(LSTM)** to predict the future price of stock. I have already trained a model and saved the LSTM Model
in my root directory And we are Accessing The model using Keras Library.Here I have also included Streamlit Library so that we can Easily visualise the predicted and actual Model on the webapp

Here the default Stock setup is SBIN.NS if you want to change keyword then Choose another from yahoo finanace.

Dependancies 
-------------
In order to run this code you're supposed to have **streamlit**,**numpy**,**python pandas** and **matplotlib** libary installed
on your machine, you can just use *pip* command to this.

```bash
-> pip install pandas
-> pip install streamlit
-> pip install keras
-> pip install matplotlib
-> pip install numpy
```
NB: If you have already installed tensorflow on your machine then simply 
>From tensorflow import keras

Running the code 
-----------------
In order to use this code, firstly clone the repo using **git** or download the zip file manually

```bash
$-> git clone https://github.com/RENJITHVS/python-stock-prediction
$-> cd python-stock-prediction
$ python-stock-prediction --> streamlit run app.py
```


