#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:29:24 2024

@author: jczars
"""
# Importing the Required Library
import pywhatkit
import datetime

def send(msn):
    hora_atual = datetime.datetime.now()
    
    
    hora=hora_atual.hour
    minu=hora_atual.minute+2
    print(hora,':',minu)
    
    pywhatkit.sendwhatmsg(
        phone_no="+5589999224402", 
        message="Teste finalizado "+msn+'-'+str(hora)+":"+str(minu), 
        time_hour=hora_atual.hour,
        time_min=minu
    )

if __name__=="__main__": 
    message='[INFO] erro '
    send(message)