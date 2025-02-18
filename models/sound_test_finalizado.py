#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:20:17 2024

@author: jczars
"""

import os

import time

def beep(times, sms=""):
    if sms=="":
        for x in range(times):  
            time.sleep(1)
            print('\a')
            os.system('spd-say "Teste  finalizado"')
    else:
        for x in range(times):  
            time.sleep(1)
            print('\a')
            os.system('spd-say "'+sms+'"')  

if __name__=="__main__":
    beep(10)

