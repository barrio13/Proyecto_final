import os
import smtplib
import pandas as pd
import numpy as np
from email.message import EmailMessage

data = pd.read_csv('C:\\Users\\guill\\OneDrive\\Desktop\\main\\googleform.csv')
df_filtrado = data[data.iloc[:, 4].str.contains('NVDA', na=False)]

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





