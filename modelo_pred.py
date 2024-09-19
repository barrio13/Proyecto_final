import os
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


# Cargar el modelo
loaded_model = load_model('C:\\Users\\guill\\OneDrive\\Desktop\\main\\model_nvda')
print("Modelo cargado exitosamente.")

p = 'C:\\Users\\guill\\OneDrive\\Desktop\\main\\sp500_stocks.csv'
symbol = 'NVDA'

# Cargar y preparar los datos
def prep_datos(p,symbol):
 data1 = pd.read_csv(p, sep=',', decimal='.')
 data = data1[data1['Symbol'] == symbol]
 data['Date'] = pd.to_datetime(data['Date'])
 data = data.sort_values(by='Date')
 data = data.drop(['Symbol', 'Adj Close', 'High', 'Low', 'Open', 'Volume'], axis=1)
 data.set_index('Date', inplace=True)
 return data

data = prep_datos(p,symbol)

# Normalizamos la columna Close
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Predicción basada en los últimos 60 días
def predecir_siguiente_dia(model, datos, scaler):
    ult_dias = datos[-60:]
    ult_dias = ult_dias.reshape((1, 60, 1))
    pred_close_scaled = model.predict(ult_dias)
    return scaler.inverse_transform(pred_close_scaled)

pred_close = predecir_siguiente_dia(loaded_model, scaled_data, scaler)
print(f'Predicted Close for day 1: {pred_close[0][0]}')

# Agregar la predicción al DataFrame
pred_fecha = data.index[-1] + pd.Timedelta(days=1)
pred_close_df = pd.DataFrame({'Close': pred_close.flatten()}, index=[pred_fecha])
data = pd.concat([data, pred_close_df])

# Normalizar y predecir el siguiente día nuevamente
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
pred_close1 = predecir_siguiente_dia(loaded_model, scaled_data, scaler)
print(f'Predicted Close for day 2: {pred_close1[0][0]}')

# Agregar la predicción al DataFrame
pred_fecha1 = data.index[-1] + pd.Timedelta(days=1)
pred_close_df1 = pd.DataFrame({'Close': pred_close1.flatten()}, index=[pred_fecha])
data = pd.concat([data, pred_close_df1])

# Normalizar y predecir el siguiente día nuevamente
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
pred_close2 = predecir_siguiente_dia(loaded_model, scaled_data, scaler)
print(f'Predicted Close for day 3: {pred_close2[0][0]}')

def tendencia(a,b):
    if a-b <= 0:
        return 'negativa'
    else:
        return 'positiva'
    
# Extraer el último valor real del cierre antes de realizar predicciones
ultimo_valor_real = data['Close'].iloc[-3]
print('ultimo valor real: ', ultimo_valor_real)

# Extraer el último valor predicho
ultimo_valor_predicho = pred_close2[0][0]

# Calcular la tendencia
t1=tendencia(ultimo_valor_predicho, ultimo_valor_real)

# Crear DataFrame para guardar en CSV
resultado_df = pd.DataFrame({
    'Empresa': [symbol],
    'Último Valor Predicho': [ultimo_valor_predicho],
    'Tendencia': [t1]
})

# Nombre del archivo CSV
csv_path = 'C:\\Users\\guill\\OneDrive\\Desktop\\main\\resultado_prediccion.csv'

# Guardar el DataFrame actualizado en el archivo CSV
if os.path.exists(csv_path):
    resultado_existente = pd.read_csv(csv_path)
    resultado_df = pd.concat([resultado_existente, resultado_df], ignore_index=True)


resultado_df.to_csv(csv_path, index=False)

print(f'Resultados añadidos al archivo {csv_path}')