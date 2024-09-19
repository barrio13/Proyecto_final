import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras import Sequential
from keras.regularizers import l2
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from keras.models import save_model
from keras.models import load_model

p = 'C:\\Users\\guill\\OneDrive\\Desktop\\gcp_proy\\sp500_stocks.csv'
symbol = 'NVDA'

# Cargar y preparar los datos
def prep_datos(p,symbol):
 data1 = pd.read_csv(p, sep=',', decimal='.')
 data = data1[data1['Symbol'] == symbol]
 data['Date'] = pd.to_datetime(data['Date'])
 data = data.sort_values(by='Date')
 data = data.drop(['Symbol', 'Adj Close', 'High', 'Low', 'Open', 'Volume'], axis=1)
 data.set_index('Date', inplace=True)
 data = data[(data.index >= '2020-01-01') & (data.index <= '2024-12-31')]
 return data

data = prep_datos(p,symbol)

# Normalizamos la columna Close
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Separamos los datos en secuencias de x días para entrenar el modelo.
def secuencias(data, num_dias):
    X, y = [], []
    for i in range(num_dias, len(data)):
        X.append(data[i-num_dias:i,0])
        y.append(data[i,0])
    return np.array(X),np.array(y)

num_dias = 60
X, y = secuencias(scaled_data, num_dias)

# Determinamos los tamaños de cada conjunto
train_size = int(X.shape[0] * 0.70)  
validation_size = int(X.shape[0] * 0.15)

# Dividimos los datos
X_train = X[:train_size]
y_train = y[:train_size]

X_validation = X[train_size:train_size + validation_size]
y_validation = y[train_size:train_size + validation_size]

X_test = X[train_size + validation_size:]
y_test = y[train_size + validation_size:]

print('X_train:',X_train.shape)
print('X_val:',X_validation.shape)
print('X_test:',X_test.shape)


# Construimos el modelo
model1 = Sequential()
model1.add(LSTM(32, input_shape=(X_train.shape[1], 1),return_sequences=True, kernel_regularizer=l2(0.001)))
model1.add(Dropout(0.2))
model1.add(LSTM(16))
model1.add(Dense(1))
optimizer = Adam(learning_rate=0.001)
model1.compile(optimizer='adam', loss='mse')
model1.summary()

# Reshape data
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_validation = X_validation.reshape((X_validation.shape[0], X_validation.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Callbacks para mejorar el entrenamiento
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Entrenamiento del modelo
history1 = model1.fit(X_train, y_train, epochs=80, validation_data=(X_validation, y_validation), callbacks=[reduce_lr, early_stop])

# Guarda el modelo en un archivo
# model1.save('C:\\Users\\guill\\OneDrive\\Desktop\\main\\model_nvda')

# # Cargar el modelo
# loaded_model = load_model('C:\\Users\\guill\\OneDrive\\Desktop\\main\\model_nvda')
# print("Modelo cargado exitosamente.")

# Graficar los datos filtrados
plt.plot(data.index, data['Close'])
plt.title(f'{symbol} Stock Prices (2020 - 2024)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.show()


# Filtrar las fechas para cada conjunto
train_data_len = train_size
val_data_len = train_size + validation_size
test_data_len = train_size + validation_size + len(X_test)

data_train = data.iloc[:train_data_len]
data_val = data.iloc[train_data_len:val_data_len]
data_test = data.iloc[val_data_len:test_data_len]

# Graficar las tres partes(Train, Test, Validation)
plt.figure(figsize=(10, 6))
plt.plot(data_train.index, data_train['Close'], label='Train', color='blue')
plt.plot(data_val.index, data_val['Close'], label='Validation', color='orange')
plt.plot(data_test.index, data_test['Close'], label='Test', color='green')

# Agregar títulos y leyendas
plt.title(f'{symbol} Stock Prices - Train, Validation y Test')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()

# Mostrar el gráfico
plt.show()

# Visualización de la función Loss
plt.plot(history1.history['loss'],label='loss')
plt.plot(history1.history['val_loss'],label='val_loss')

# Agregar títulos y leyendas
plt.title(f'{symbol} Loss Function')
plt.xlabel('Epochs')
plt.legend()
plt.show()

# Evaluación en el conjunto de prueba
test = model1.evaluate(X_test, y_test)
print(f'Test loss: {test}')