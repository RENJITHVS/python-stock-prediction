import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as pt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
from pandas_datareader._utils import RemoteDataError

pt.style.use('fivethirtyeight')


start = "2012-01-01"
end= "2022-05-08"

st.title('Stock Trend prediction')

user_input = st.text_input("enter the stock Name")

try:
    df = data.DataReader(user_input,'yahoo',start, end)
except RemoteDataError:
    pass
finally:
    df = data.DataReader('SBIN.NS','yahoo',start, end)
    
#describing data
st.subheader("Data from 2010-2022")
st.write(df.describe())

#Visualisation
st.subheader('Closing Price vs time chart')
# fig = pt.figure(figsize=(12,6))
# pt.plot(df.Close)
# st.pyplot(fig)
fig=pt.figure(figsize=(16,8))
pt.title('Close Price History')
pt.xlabel('Date',fontsize=18)
pt.ylabel('Close Price INR',fontsize=18)
pt.plot(df['Close'])
st.pyplot(fig)


data=df.filter(['Close'])
dataset=data.values
training_data_len= math.ceil(len(dataset) * .8)

from sklearn.preprocessing import MinMaxScaler
#scale the data
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)

#Create training data set
train_data=scaled_data[0:training_data_len,:]

# load model
model = load_model('keras_model.h5')

#  Testing part

#create the testing data set
#create a new array containing scaled values from index 1543 to 2003
test_data=scaled_data[training_data_len - 60:,:]
#create the data sets x_test and y_test
x_test=[]
y_test=dataset[training_data_len:,:]
for i in range(60,len(test_data)):
  x_test.append(test_data[i-60:i,0])
  
#convert the data to a numpy array
x_test=np.array(x_test)
#reshape
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

#get the models predicted price values
predictions=model.predict(x_test)
predictions=scaler.inverse_transform(predictions)

#Get the root mean squared error(RMSE)
rmse=np.sqrt( np.mean((predictions - y_test)**2))

train=data[:training_data_len]
valid=data[training_data_len:]
valid['Predictions']=predictions

#visualize the data
st.subheader('Predicted Closing Price vs time chart')
fig3 = pt.figure(figsize=(16,8))
pt.title('Model')
pt.xlabel('Date',fontsize=18)
pt.ylabel('Close Price INR',fontsize=18)
pt.plot(train['Close'])
pt.plot(valid[['Close','Predictions']])
pt.legend(['Train','Val','Predictions'],loc='lower right')
st.pyplot(fig3)
