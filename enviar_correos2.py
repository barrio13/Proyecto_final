import gspread
import pandas as pd
from google.oauth2.service_account import Credentials
import os
import smtplib
import pandas as pd
import numpy as np
from email.message import EmailMessage

# Autenticación y autorización
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = Credentials.from_service_account_file('C:\\Users\\guill\\OneDrive\\Desktop\\gcp_proy\\secret.json', scopes=scope)
client = gspread.authorize(creds)

# Abrir la hoja de cálculo
spreadsheet = client.open("KeepCodingSheet")
sheet = spreadsheet.sheet1  # Especificar la hoja por nombre o índice

# Recopilar información de ciertas columnas
column_data_email = sheet.col_values(4)[1:] # columna del email (se quita el primer item que es el titulo de la columna)
column_data_company = sheet.col_values(5)[1:] # columna de la compañia (se quita el primer item que es el titulo de la columna)

# Se almacena mail y preferencia del usuario en una lista de listas
info = []
for i in range(len(column_data_email)):
    info.append([column_data_email[i], column_data_company[i]])

# Devolver la información en df
# Convertir la lista de listas a un DataFrame
df = pd.DataFrame(info, columns=['Email', 'Empresas preferidas'])
print(df)

df_filtrado = df[df['Empresas preferidas'].str.contains('NVDA', na=False)]

data_modelo = pd.read_csv('C:\\Users\\guill\\OneDrive\\Desktop\\main\\resultado_prediccion.csv')
tendencia = data_modelo['Tendencia'].tolist()

lista_emails = df_filtrado['Email'].tolist()
email_adress = 'keepproyecto@gmail.com'
email_password = os.environ.get('mail_pass')
print(lista_emails)

msg = EmailMessage()
msg['Subject']= 'Predicción de acciones'
msg['From'] = email_adress
msg['To'] = lista_emails
msg.set_content(f'La tendencia es \n{tendencia[0]}')

with smtplib.SMTP('smtp.gmail.com',587) as smtp:
    smtp.ehlo()
    smtp.starttls()
    smtp.ehlo()
    smtp.login(email_adress,email_password)
    smtp.send_message(msg)
