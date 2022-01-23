# %% [code] {"execution":{"iopub.status.busy":"2021-11-30T03:17:29.403185Z","iopub.execute_input":"2021-11-30T03:17:29.40415Z","iopub.status.idle":"2021-11-30T03:17:30.268141Z","shell.execute_reply.started":"2021-11-30T03:17:29.40404Z","shell.execute_reply":"2021-11-30T03:17:30.267346Z"}}
import numpy as np
import pandas
import seaborn
import matplotlib.pyplot as plt

# %% [code] {"execution":{"iopub.status.busy":"2021-11-30T03:17:30.269801Z","iopub.execute_input":"2021-11-30T03:17:30.270128Z","iopub.status.idle":"2021-11-30T03:17:38.23477Z","shell.execute_reply.started":"2021-11-30T03:17:30.270089Z","shell.execute_reply":"2021-11-30T03:17:38.233923Z"}}
df = pandas.read_csv('../input/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv')

# %% [code] {"execution":{"iopub.status.busy":"2021-11-30T03:17:38.236091Z","iopub.execute_input":"2021-11-30T03:17:38.236415Z","iopub.status.idle":"2021-11-30T03:17:38.262738Z","shell.execute_reply.started":"2021-11-30T03:17:38.236375Z","shell.execute_reply":"2021-11-30T03:17:38.261896Z"}}
df.head()

# %% [code] {"execution":{"iopub.status.busy":"2021-11-30T03:17:38.264358Z","iopub.execute_input":"2021-11-30T03:17:38.264663Z","iopub.status.idle":"2021-11-30T03:17:38.469334Z","shell.execute_reply.started":"2021-11-30T03:17:38.264613Z","shell.execute_reply":"2021-11-30T03:17:38.468366Z"}}
from tabulate import tabulate

info = [[col, df[col].count(), df[col].max(), df[col].min()] for col in df.columns]
print(tabulate(info, headers=['Feature', 'Count', 'Max', 'Min'], tablefmt='orgtbl'))

# %% [markdown]
# # Exploratory Data analysis Part 2

# %% [code] {"execution":{"iopub.status.busy":"2021-11-30T03:17:38.470767Z","iopub.execute_input":"2021-11-30T03:17:38.471063Z","iopub.status.idle":"2021-11-30T03:17:38.560568Z","shell.execute_reply.started":"2021-11-30T03:17:38.471026Z","shell.execute_reply":"2021-11-30T03:17:38.559738Z"}}
print(df.isna().sum())

# %% [markdown]
# There are more than **1 million** unrecorded timestamps.

# %% [code] {"execution":{"iopub.status.busy":"2021-11-30T03:17:38.56194Z","iopub.execute_input":"2021-11-30T03:17:38.562343Z","iopub.status.idle":"2021-11-30T03:17:38.906164Z","shell.execute_reply.started":"2021-11-30T03:17:38.562299Z","shell.execute_reply":"2021-11-30T03:17:38.90527Z"}}
df = df.dropna()

# %% [code] {"execution":{"iopub.status.busy":"2021-11-30T03:17:38.90737Z","iopub.execute_input":"2021-11-30T03:17:38.908088Z","iopub.status.idle":"2021-11-30T03:17:38.972334Z","shell.execute_reply.started":"2021-11-30T03:17:38.908055Z","shell.execute_reply":"2021-11-30T03:17:38.971479Z"}}
print('total missing values : ' + str(df.isna().sum().sum()))

# %% [code] {"execution":{"iopub.status.busy":"2021-11-30T03:17:38.973714Z","iopub.execute_input":"2021-11-30T03:17:38.974283Z","iopub.status.idle":"2021-11-30T03:17:38.992039Z","shell.execute_reply.started":"2021-11-30T03:17:38.974239Z","shell.execute_reply":"2021-11-30T03:17:38.991244Z"}}
df = df[df['Timestamp'] > (df['Timestamp'].max() - 650000)]
print(df['Timestamp'].max())

# %% [code] {"execution":{"iopub.status.busy":"2021-11-30T03:17:38.993993Z","iopub.execute_input":"2021-11-30T03:17:38.994283Z","iopub.status.idle":"2021-11-30T03:17:38.999615Z","shell.execute_reply.started":"2021-11-30T03:17:38.994253Z","shell.execute_reply":"2021-11-30T03:17:38.998669Z"}}
df = df.reset_index(drop=True)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-30T03:17:39.000492Z","iopub.execute_input":"2021-11-30T03:17:39.001038Z","iopub.status.idle":"2021-11-30T03:17:39.020188Z","shell.execute_reply.started":"2021-11-30T03:17:39.001006Z","shell.execute_reply":"2021-11-30T03:17:39.019449Z"}}
df.head()

# %% [code] {"execution":{"iopub.status.busy":"2021-11-30T03:17:39.021103Z","iopub.execute_input":"2021-11-30T03:17:39.021714Z","iopub.status.idle":"2021-11-30T03:17:40.638241Z","shell.execute_reply.started":"2021-11-30T03:17:39.021668Z","shell.execute_reply":"2021-11-30T03:17:40.635755Z"}}
df.hist(figsize=(10, 9))
plt.savefig('histogram.png')
plt.show();
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2021-11-30T03:17:40.639604Z","iopub.execute_input":"2021-11-30T03:17:40.640238Z","iopub.status.idle":"2021-11-30T03:17:41.472521Z","shell.execute_reply.started":"2021-11-30T03:17:40.640193Z","shell.execute_reply":"2021-11-30T03:17:41.471922Z"}}
plt.figure(figsize=(10, 10))
plt.savefig('Correlation.png')
plt.show()
m = df.corr()
seaborn_plot = seaborn.heatmap(m, vmin=-1.0, annot=True, square=True)
seaborn_plot.figure.savefig("output.png")

# %% [code] {"execution":{"iopub.status.busy":"2021-11-30T03:17:41.473719Z","iopub.execute_input":"2021-11-30T03:17:41.474495Z","iopub.status.idle":"2021-11-30T03:17:41.480173Z","shell.execute_reply.started":"2021-11-30T03:17:41.474459Z","shell.execute_reply":"2021-11-30T03:17:41.479357Z"}}
df = df.drop(['Timestamp', 'Low', 'High', 'Volume_(BTC)', 'Weighted_Price'], axis=1)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-30T03:17:41.481191Z","iopub.execute_input":"2021-11-30T03:17:41.481399Z","iopub.status.idle":"2021-11-30T03:17:41.493617Z","shell.execute_reply.started":"2021-11-30T03:17:41.481376Z","shell.execute_reply":"2021-11-30T03:17:41.493071Z"}}
info = [[col, df[col].count(), df[col].max(), df[col].min()] for col in df.columns]
print(tabulate(info, headers=['Feature', 'Count', 'Max', 'Min'], tablefmt='orgtbl'))

# %% [code] {"execution":{"iopub.status.busy":"2021-11-30T03:17:41.495378Z","iopub.execute_input":"2021-11-30T03:17:41.495775Z","iopub.status.idle":"2021-11-30T03:17:42.016405Z","shell.execute_reply.started":"2021-11-30T03:17:41.495737Z","shell.execute_reply":"2021-11-30T03:17:42.015745Z"}}
plt.figure(figsize=(10, 10))
plt.subplot(2, 1, 1)
plt.plot(df['Open'].values[df.shape[0] - 500:df.shape[0]])
plt.xlabel('Time period')
plt.ylabel('Opening price')
plt.title('Opening price of Bitcoin for last 500 timestamps')

plt.subplot(2, 1, 2)
plt.plot(df['Volume_(Currency)'].values[df.shape[0] - 500:df.shape[0]])
plt.xlabel('Time period')
plt.ylabel('Volume Traded')
plt.title('Volume traded of Bitcoin for last 500 timestamps')
plt.savefig('my_image.png')
plt.show()

# %% [markdown]
# # Creating  the arrays

# %% [code] {"execution":{"iopub.status.busy":"2021-11-30T03:17:42.060292Z","iopub.execute_input":"2021-11-30T03:17:42.061135Z","iopub.status.idle":"2021-11-30T03:17:42.066362Z","shell.execute_reply.started":"2021-11-30T03:17:42.061098Z","shell.execute_reply":"2021-11-30T03:17:42.065584Z"}}
a = np.array(df.drop(['Close'], axis=1))
b = np.array(df['Close'])

# %% [code] {"execution":{"iopub.status.busy":"2021-11-30T03:17:42.335214Z","iopub.execute_input":"2021-11-30T03:17:42.335491Z","iopub.status.idle":"2021-11-30T03:17:42.34165Z","shell.execute_reply.started":"2021-11-30T03:17:42.335465Z","shell.execute_reply":"2021-11-30T03:17:42.34084Z"}}
print(a.shape)
print(b.shape)

# %% [markdown]
# # Data Scaling
#

# %% [code] {"execution":{"iopub.status.busy":"2021-11-30T03:18:01.3685Z","iopub.execute_input":"2021-11-30T03:18:01.36881Z","iopub.status.idle":"2021-11-30T03:18:01.374849Z","shell.execute_reply.started":"2021-11-30T03:18:01.36878Z","shell.execute_reply":"2021-11-30T03:18:01.373974Z"}}
from sklearn.preprocessing import StandardScaler

a = StandardScaler().fit_transform(a)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-30T03:18:03.900523Z","iopub.execute_input":"2021-11-30T03:18:03.901034Z","iopub.status.idle":"2021-11-30T03:18:03.907213Z","shell.execute_reply.started":"2021-11-30T03:18:03.901004Z","shell.execute_reply":"2021-11-30T03:18:03.906517Z"}}
t = np.reshape(b, (-1, 1))
b = StandardScaler().fit_transform(t)
b = b.reshape(-1)

# %% [markdown]
# # Creating the  time series datasets
# Considering past **500** timestamps,which are  approximately equal to 8 hours.

# %% [code] {"execution":{"iopub.status.busy":"2021-11-30T03:18:11.763603Z","iopub.execute_input":"2021-11-30T03:18:11.764359Z","iopub.status.idle":"2021-11-30T03:18:11.769155Z","shell.execute_reply.started":"2021-11-30T03:18:11.764319Z","shell.execute_reply":"2021-11-30T03:18:11.768339Z"}}
print(a.shape)
print(b.shape)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-30T03:18:12.610395Z","iopub.execute_input":"2021-11-30T03:18:12.610951Z","iopub.status.idle":"2021-11-30T03:18:12.684992Z","shell.execute_reply.started":"2021-11-30T03:18:12.610908Z","shell.execute_reply":"2021-11-30T03:18:12.684049Z"}}
size = 500
xa_temp = []
ya_temp = []
for k in range(size, a.shape[0]):
    xa_temp.append(a[k - size: k])
    ya_temp.append(b[k])
xa_temp = np.array(xa_temp)
ya_temp = np.array(ya_temp)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-30T03:18:13.235375Z","iopub.execute_input":"2021-11-30T03:18:13.2362Z","iopub.status.idle":"2021-11-30T03:18:13.241289Z","shell.execute_reply.started":"2021-11-30T03:18:13.236153Z","shell.execute_reply":"2021-11-30T03:18:13.240201Z"}}
print(xa_temp.shape)
print(ya_temp.shape)

# %% [markdown]
# # Train test split

# %% [code] {"execution":{"iopub.status.busy":"2021-11-30T03:18:14.187577Z","iopub.execute_input":"2021-11-30T03:18:14.188075Z","iopub.status.idle":"2021-11-30T03:18:14.273753Z","shell.execute_reply.started":"2021-11-30T03:18:14.188036Z","shell.execute_reply":"2021-11-30T03:18:14.272854Z"}}
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(xa_temp, ya_temp, test_size=0.2, random_state=1)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-30T03:18:15.060962Z","iopub.execute_input":"2021-11-30T03:18:15.061286Z","iopub.status.idle":"2021-11-30T03:18:15.066934Z","shell.execute_reply.started":"2021-11-30T03:18:15.061254Z","shell.execute_reply":"2021-11-30T03:18:15.06577Z"}}
print(X_train.shape)
print(y_train.shape)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-30T03:18:15.613501Z","iopub.execute_input":"2021-11-30T03:18:15.61444Z","iopub.status.idle":"2021-11-30T03:18:15.619818Z","shell.execute_reply.started":"2021-11-30T03:18:15.614395Z","shell.execute_reply":"2021-11-30T03:18:15.618885Z"}}
print(X_test.shape)
print(y_test.shape)

# %% [markdown]
# # Models (RNN and LSTM)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-30T03:18:19.758462Z","iopub.execute_input":"2021-11-30T03:18:19.759054Z","iopub.status.idle":"2021-11-30T03:18:25.061Z","shell.execute_reply.started":"2021-11-30T03:18:19.759017Z","shell.execute_reply":"2021-11-30T03:18:25.060061Z"}}
from tensorflow import keras
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.layers import Input


# %% [code] {"execution":{"iopub.status.busy":"2021-11-30T03:18:25.062716Z","iopub.execute_input":"2021-11-30T03:18:25.063009Z","iopub.status.idle":"2021-11-30T03:18:25.069973Z","shell.execute_reply.started":"2021-11-30T03:18:25.062978Z","shell.execute_reply":"2021-11-30T03:18:25.068988Z"}}
def layer(hidden1):
    model = keras.models.Sequential()

    # add input layer
    model.add(Input(shape=(500, 2,)))

    # add rnn layer
    model.add(SimpleRNN(hidden1, activation='tanh', return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # add output layer
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model


# %% [code] {"execution":{"iopub.status.busy":"2021-11-30T03:18:25.07146Z","iopub.execute_input":"2021-11-30T03:18:25.071999Z","iopub.status.idle":"2021-11-30T03:18:25.28904Z","shell.execute_reply.started":"2021-11-30T03:18:25.071965Z","shell.execute_reply":"2021-11-30T03:18:25.288167Z"}}
model = layer(10)
model.summary()

# %% [code] {"execution":{"iopub.status.busy":"2021-11-30T03:18:25.290809Z","iopub.execute_input":"2021-11-30T03:18:25.29105Z","iopub.status.idle":"2021-11-30T03:18:25.295153Z","shell.execute_reply.started":"2021-11-30T03:18:25.291023Z","shell.execute_reply":"2021-11-30T03:18:25.294535Z"}}
from tensorflow.keras.callbacks import ModelCheckpoint

checkp = ModelCheckpoint('./bit_model.h5', monitor='val_loss', save_best_only=True, verbose=1)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-30T03:18:25.411548Z","iopub.execute_input":"2021-11-30T03:18:25.41206Z","iopub.status.idle":"2021-11-30T03:18:25.416591Z","shell.execute_reply.started":"2021-11-30T03:18:25.412013Z","shell.execute_reply":"2021-11-30T03:18:25.415819Z"}}
import time

beg = time.time()

# %% [code] {"execution":{"iopub.status.busy":"2021-11-27T21:59:31.873516Z","iopub.execute_input":"2021-11-27T21:59:31.873851Z","iopub.status.idle":"2021-11-27T22:03:48.942049Z","shell.execute_reply.started":"2021-11-27T21:59:31.873813Z","shell.execute_reply":"2021-11-27T22:03:48.941411Z"}}
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test), callbacks=[checkp])

# %% [code] {"execution":{"iopub.status.busy":"2021-11-27T22:03:48.943455Z","iopub.execute_input":"2021-11-27T22:03:48.943802Z","iopub.status.idle":"2021-11-27T22:03:48.948978Z","shell.execute_reply.started":"2021-11-27T22:03:48.943761Z","shell.execute_reply":"2021-11-27T22:03:48.948095Z"}}
end = time.time()

# %% [code] {"execution":{"iopub.status.busy":"2021-11-27T22:03:48.950413Z","iopub.execute_input":"2021-11-27T22:03:48.951196Z","iopub.status.idle":"2021-11-27T22:03:49.080048Z","shell.execute_reply.started":"2021-11-27T22:03:48.951121Z","shell.execute_reply":"2021-11-27T22:03:49.079404Z"}}
from tensorflow.keras.models import load_model

model = load_model('./bit_model.h5')

# %% [code] {"execution":{"iopub.status.busy":"2021-11-27T22:03:49.082512Z","iopub.execute_input":"2021-11-27T22:03:49.083476Z","iopub.status.idle":"2021-11-27T22:03:50.969399Z","shell.execute_reply.started":"2021-11-27T22:03:49.083425Z","shell.execute_reply":"2021-11-27T22:03:50.968651Z"}}
pred = model.predict(X_test)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-27T22:03:50.972193Z","iopub.execute_input":"2021-11-27T22:03:50.972547Z","iopub.status.idle":"2021-11-27T22:03:50.978637Z","shell.execute_reply.started":"2021-11-27T22:03:50.972506Z","shell.execute_reply":"2021-11-27T22:03:50.977936Z"}}
print(pred.shape)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-27T22:03:50.979574Z","iopub.execute_input":"2021-11-27T22:03:50.980268Z","iopub.status.idle":"2021-11-27T22:03:50.991626Z","shell.execute_reply.started":"2021-11-27T22:03:50.980233Z","shell.execute_reply":"2021-11-27T22:03:50.990768Z"}}
pred = pred.reshape(-1)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-27T22:03:50.994419Z","iopub.execute_input":"2021-11-27T22:03:50.995098Z","iopub.status.idle":"2021-11-27T22:03:51.007305Z","shell.execute_reply.started":"2021-11-27T22:03:50.995053Z","shell.execute_reply":"2021-11-27T22:03:51.00615Z"}}
from sklearn.metrics import mean_squared_error

print('MSE : ' + str(mean_squared_error(y_test, pred)))

# %% [code] {"execution":{"iopub.status.busy":"2021-11-27T22:03:51.008734Z","iopub.execute_input":"2021-11-27T22:03:51.009038Z","iopub.status.idle":"2021-11-27T22:03:51.417723Z","shell.execute_reply.started":"2021-11-27T22:03:51.008936Z","shell.execute_reply":"2021-11-27T22:03:51.416849Z"}}
plt.figure(figsize=(15, 8))
plt.plot(y_test[2040:2060])
plt.plot(pred[2040:2060])
plt.xlabel('Time', fontsize=20)
plt.ylabel('Price', fontsize=20)
plt.title('Closing Price vs Time (using SimpleRNN)')
plt.legend(['Actual price', 'Predicted price'])
plt.savefig('RNN')
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2021-11-27T22:03:51.419606Z","iopub.execute_input":"2021-11-27T22:03:51.42041Z","iopub.status.idle":"2021-11-27T22:03:51.426705Z","shell.execute_reply.started":"2021-11-27T22:03:51.420366Z","shell.execute_reply":"2021-11-27T22:03:51.425725Z"}}
print('Time taken for SimpleRNN model to learn : ' + str(end - beg) + ' sec.')


# %% [code] {"execution":{"iopub.status.busy":"2021-11-27T22:03:51.428073Z","iopub.execute_input":"2021-11-27T22:03:51.428397Z","iopub.status.idle":"2021-11-27T22:03:51.436885Z","shell.execute_reply.started":"2021-11-27T22:03:51.428365Z","shell.execute_reply":"2021-11-27T22:03:51.436309Z"}}
def layerls(hidden1):
    model = keras.models.Sequential()

    # add input layer
    model.add(Input(shape=(500, 2,)))

    # add rnn layer
    model.add(LSTM(hidden1, activation='tanh', return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # add output layer
    model.add(Dense(1, activation='linear'))

    model.compile(loss="mean_squared_error", optimizer='adam')

    return model


# %% [code] {"execution":{"iopub.status.busy":"2021-11-27T22:03:51.438091Z","iopub.execute_input":"2021-11-27T22:03:51.438684Z","iopub.status.idle":"2021-11-27T22:03:51.763635Z","shell.execute_reply.started":"2021-11-27T22:03:51.438645Z","shell.execute_reply":"2021-11-27T22:03:51.76267Z"}}
model = layerls(256)
model.summary()

# %% [code] {"execution":{"iopub.status.busy":"2021-11-27T22:03:51.764733Z","iopub.execute_input":"2021-11-27T22:03:51.76497Z","iopub.status.idle":"2021-11-27T22:03:51.770498Z","shell.execute_reply.started":"2021-11-27T22:03:51.764943Z","shell.execute_reply":"2021-11-27T22:03:51.769482Z"}}
checkp = ModelCheckpoint('./bit_model_lstm.h5', monitor='val_loss', save_best_only=True, verbose=1)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-27T22:03:51.772146Z","iopub.execute_input":"2021-11-27T22:03:51.772535Z","iopub.status.idle":"2021-11-27T22:03:51.781211Z","shell.execute_reply.started":"2021-11-27T22:03:51.772494Z","shell.execute_reply":"2021-11-27T22:03:51.780569Z"}}
beg = time.time()

# %% [code] {"execution":{"iopub.status.busy":"2021-11-27T22:22:24.940416Z","iopub.execute_input":"2021-11-27T22:22:24.940797Z","iopub.status.idle":"2021-11-27T22:41:20.993523Z","shell.execute_reply.started":"2021-11-27T22:22:24.940758Z","shell.execute_reply":"2021-11-27T22:41:20.991255Z"}}
# It stopped early due to avaliable tpu quota,mse value changes due to different value initialization
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test), callbacks=[checkp])

# %% [code] {"execution":{"iopub.status.busy":"2021-11-27T22:41:26.000062Z","iopub.execute_input":"2021-11-27T22:41:26.00039Z","iopub.status.idle":"2021-11-27T22:41:26.004601Z","shell.execute_reply.started":"2021-11-27T22:41:26.000353Z","shell.execute_reply":"2021-11-27T22:41:26.0037Z"}}
end = time.time()

# %% [code] {"execution":{"iopub.status.busy":"2021-11-27T22:41:26.973938Z","iopub.execute_input":"2021-11-27T22:41:26.974598Z","iopub.status.idle":"2021-11-27T22:41:49.0189Z","shell.execute_reply.started":"2021-11-27T22:41:26.974558Z","shell.execute_reply":"2021-11-27T22:41:49.018219Z"}}
pred = model.predict(X_test)
z = pred

# %% [code] {"execution":{"iopub.status.busy":"2021-11-27T22:41:49.020739Z","iopub.execute_input":"2021-11-27T22:41:49.021215Z","iopub.status.idle":"2021-11-27T22:41:49.025842Z","shell.execute_reply.started":"2021-11-27T22:41:49.021169Z","shell.execute_reply":"2021-11-27T22:41:49.024835Z"}}
z = z.reshape(-1)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-27T22:41:49.026803Z","iopub.execute_input":"2021-11-27T22:41:49.027619Z","iopub.status.idle":"2021-11-27T22:41:49.039173Z","shell.execute_reply.started":"2021-11-27T22:41:49.027585Z","shell.execute_reply":"2021-11-27T22:41:49.038233Z"}}
print('MSE : ' + str(mean_squared_error(y_test, z)))

# %% [code] {"execution":{"iopub.status.busy":"2021-11-27T22:41:49.041457Z","iopub.execute_input":"2021-11-27T22:41:49.041835Z","iopub.status.idle":"2021-11-27T22:41:49.388524Z","shell.execute_reply.started":"2021-11-27T22:41:49.041777Z","shell.execute_reply":"2021-11-27T22:41:49.387393Z"}}
plt.figure(figsize=(10, 7))
plt.plot(y_test[2040:2060])
plt.plot(pred[2040:2060])
plt.xlabel('Time', fontsize=20)
plt.ylabel('Price', fontsize=20)
plt.title('Closing Price vs Time (using LSTM)')
plt.legend(['Actual price', 'Predicted price'])
plt.savefig('LSTM')
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2021-11-27T22:43:17.516666Z","iopub.execute_input":"2021-11-27T22:43:17.516992Z","iopub.status.idle":"2021-11-27T22:43:17.523192Z","shell.execute_reply.started":"2021-11-27T22:43:17.516952Z","shell.execute_reply":"2021-11-27T22:43:17.521951Z"}}
print('Time taken by LSTM to learn : ' + str(end - beg))

# %% [code]


# %% [code]


# %% [code]


# %% [code]


# %% [code]


# %% [code]


# %% [code]


# %% [code]


# %% [code]


# %% [code]


# %% [code]


# %% [code]
