import cv2
import numpy as np
from time import sleep

from tflite_runtime.interpreter import Interpreter
#from tensorflow.keras.models import load_model
#from keras.models import load_model

import pyautogui
import keyboard

import WebcamModule as wM
import DataCollectionModule as dcM


import pandas as pd
import os
from datetime import datetime


import tkinter as tk

import RPi.GPIO as GPIO
import time
import sys

OutPut_Pins = [15 , 16 , 18 , 19 ,21 , 31 , 32 , 33 , 35 , 36 ]


        
#######################################
steeringSen = 1 # Steering Sensitivity
scale= 21  # Steering scale to middel
SafeSpace = 40
back = 50
model_path = 'model_v3.tflite'
image_path = 'img-52.jpg' #image example


######################################
# Load model (interpreter)
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

#print(input_details)

# Make prediction from model
def predict(img,input_details , output_details):
    #img = np.float32(mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1], 1))
    interpreter.set_tensor(input_details[0]['index'], np.float32(img))
    interpreter.invoke()
    # Obtain results and map them to the classes
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]  
    return(predictions)



def mask2(image):
    original = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([22, 93, 0], dtype="uint8")
    upper = np.array([45, 255, 255], dtype="uint8")
    mask = cv2.inRange(image, lower, upper)

    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(original, (x, y), (x + w, y + h), (36,255,12), 2)

    return original


def preProcess(img):
    #cv2.imshow("Input",img)
    img = img[24:120, :, :]
    img = mask2(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return(img)



######################################################
def init():
    GPIO.setmode(GPIO.BOARD)		#set pin numbering system
    GPIO.setwarnings(False)			#disable warnings


    for Item in OutPut_Pins:
        GPIO.setup(Item,GPIO.OUT)
        
def StopAllProcess():

    init()
    GPIO.output(35 , 1)
    GPIO.output(36 , 1)
    GPIO.output(15 , 1)
    GPIO.output(16 , 1)
    GPIO.output(18 , 1)
    GPIO.output(19 , 1)
    
    
def altra():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)
    #set GPIO Pins
    GPIO_TRIGGER1 = 3
    GPIO_ECHO1 = 5
    GPIO_TRIGGER2 = 7
    GPIO_ECHO2 = 8
    GPIO_TRIGGER3 = 10
    GPIO_ECHO3 = 11
    GPIO_TRIGGER4 = 12
    GPIO_ECHO4 = 13
     
    #set GPIO direction (IN / OUT)
    GPIO.setup(GPIO_TRIGGER1, GPIO.OUT)
    GPIO.setup(GPIO_ECHO1, GPIO.IN)
    GPIO.setup(GPIO_TRIGGER2, GPIO.OUT)
    GPIO.setup(GPIO_ECHO2, GPIO.IN)
    GPIO.setup(GPIO_TRIGGER3, GPIO.OUT)
    GPIO.setup(GPIO_ECHO3, GPIO.IN)
    GPIO.setup(GPIO_TRIGGER4, GPIO.OUT)
    GPIO.setup(GPIO_ECHO4, GPIO.IN)
    GPIO.output(GPIO_TRIGGER1, False)
    GPIO.output(GPIO_TRIGGER2, False)
    GPIO.output(GPIO_TRIGGER3, False)
    GPIO.output(GPIO_TRIGGER4, False)

    print ("sensor1")
    cv2.waitKey(50)
    GPIO.output(GPIO_TRIGGER1, True)
    cv2.waitKey(50)
    GPIO.output(GPIO_TRIGGER1, False)
    while GPIO.input(GPIO_ECHO1) == 0:
        StartTime = time.time()
    while GPIO.input(GPIO_ECHO1) == 1:
        StopTime = time.time()  
    TimeElapsed = StopTime - StartTime
    distance = TimeElapsed * 17150
    distance_Forward = round(distance, 2)
    print ("sensor1 :" ,distance_Forward, "cm")
    while(distance_Forward < SafeSpace):
        StopAllProcess()
        print("Warning Left.....")
        
        cv2.waitKey(500)
        m_left(40)
        cv2.waitKey(100)
        m_back(back)
        cv2.waitKey(100)
        m_right(10)
        
        
        cv2.waitKey(100)
        GPIO.output(GPIO_TRIGGER1, True)
        cv2.waitKey(50)
        GPIO.output(GPIO_TRIGGER1, False)
        while GPIO.input(GPIO_ECHO1) == 0:
            StartTime = time.time()
        while GPIO.input(GPIO_ECHO1) == 1:
            StopTime = time.time()  
        TimeElapsed = StopTime - StartTime
        distance = TimeElapsed * 17150
        distance_Forward = round(distance, 2)
#         response = requests.get( BaseURL + "SendData?Sensor_1=" + str(distance_Forward) + "&Sensor_2=0&Sensor_3=0&Sensor_4=0")
#     if distance_Forward < Safe_Space:
#         Move('s');
#         print("Sensor_1 Warnning.....")
#         for i in range(20):
#             SelfDriveCar(0)
#         for x in range(20):
#             Move("B1")
            
    print ("sensor2")
    cv2.waitKey(50)
    GPIO.output(GPIO_TRIGGER2, True)
    cv2.waitKey(50)
    GPIO.output(GPIO_TRIGGER2, False)
    while GPIO.input(GPIO_ECHO2) == 0:
        StartTime = time.time()
    while GPIO.input(GPIO_ECHO2) == 1:
        StopTime = time.time()
    TimeElapsed = StopTime - StartTime
    distance = TimeElapsed * 17150
    distance_Back = round(distance, 2)
    print ("sensor2 :" ,distance_Back, "cm")
    while(distance_Back < SafeSpace):
        StopAllProcess()
        print("Warning Forward.....")
        
        
        cv2.waitKey(500)
        m_right(40)
        cv2.waitKey(100)
        m_back(back)
        cv2.waitKey(100)
        m_left(10)
        
        
        cv2.waitKey(100)
        GPIO.output(GPIO_TRIGGER2, True)
        cv2.waitKey(50)
        GPIO.output(GPIO_TRIGGER2, False)
        while GPIO.input(GPIO_ECHO2) == 0:
            StartTime = time.time()
        while GPIO.input(GPIO_ECHO2) == 1:
            StopTime = time.time()
        TimeElapsed = StopTime - StartTime
        distance = TimeElapsed * 17150
        distance_Back = round(distance, 2)
#         response = requests.get( BaseURL + "SendData?Sensor_2=" +str(distance_Back) + "&Sensor_1=0&Sensor_3=0&Sensor_4=0")

#     if distance_Back < Safe_Space:
#         Move('s');
#         print("Sensor_2 Warnning.....")
#         for i in range(20):
#             SelfDriveCar(0)
#         for x in range(20):
#             Move("F1")

    print ("sensor3")
    cv2.waitKey(100)
    GPIO.output(GPIO_TRIGGER3, True)
    cv2.waitKey(100)
    GPIO.output(GPIO_TRIGGER3, False)
    while GPIO.input(GPIO_ECHO3) == 0:
        StartTime = time.time()
    while GPIO.input(GPIO_ECHO3) == 1:
        StopTime = time.time()  
    TimeElapsed = StopTime - StartTime
    distance = TimeElapsed * 17150
    distance_Right = round(distance, 2)
    print ("sensor3 :" ,distance_Right, "cm")
    while(distance_Right < SafeSpace):
        StopAllProcess()
        print("Warning Right.....")
        
        cv2.waitKey(5000)
        m_right(40)
        cv2.waitKey(100)
        m_back(back)
        cv2.waitKey(100)
        m_left(10)
        
        cv2.waitKey(100)
        GPIO.output(GPIO_TRIGGER3, True)
        cv2.waitKey(50)
        GPIO.output(GPIO_TRIGGER3, False)
        while GPIO.input(GPIO_ECHO3) == 0:
            StartTime = time.time()
        while GPIO.input(GPIO_ECHO3) == 1:
            StopTime = time.time()  
        TimeElapsed = StopTime - StartTime
        distance = TimeElapsed * 17150
        distance_Right = round(distance, 2)
#         response = requests.get( BaseURL + "SendData?Sensor_3=" + str(distance_Right) + "&Sensor_1=0&Sensor_2=0&Sensor_4=0")

#     if distance_Right < Safe_Space:
#         for i in range(20):
#             SelfDriveCar(-0.8)
#         for x in range(20):
#             Move("B1")
#     
#     
    print ("sensor4")
    cv2.waitKey(50)
    GPIO.output(GPIO_TRIGGER4, True)
    cv2.waitKey(50)
    GPIO.output(GPIO_TRIGGER4, False)
    while GPIO.input(GPIO_ECHO4) == 0:
        StartTime = time.time()
    while GPIO.input(GPIO_ECHO4) == 1:
        StopTime = time.time()
    TimeElapsed = StopTime - StartTime
    distance = TimeElapsed * 17150
    distance_Left = round(distance, 2)
    print ("sensor4 :" ,distance_Left, "cm")
    while distance_Left < SafeSpace :
        StopAllProcess()
        print("Warning Back.....")
        
        cv2.waitKey(50)
        GPIO.output(GPIO_TRIGGER4, True)
        cv2.waitKey(50)
        GPIO.output(GPIO_TRIGGER4, False)
        while GPIO.input(GPIO_ECHO4) == 0:
            StartTime = time.time()
        while GPIO.input(GPIO_ECHO4) == 1:
            StopTime = time.time()
        TimeElapsed = StopTime - StartTime
        distance = TimeElapsed * 17150
        distance_Left = round(distance, 2)
#         response = requests.get( BaseURL + "SendData?Sensor_4=" + str(distance_Left) + "&Sensor_1=0&Sensor_2=0&Sensor_3=0")
#         print(response.status_code)
        
#     if distance_Left < Safe_Space:
#         Move('s');
#         print("Sensor_4 Warnning.....")
#         for i in range(20):
#             SelfDriveCar(0.8)
#         for x in range(20):
#             Move("B1")
# 

    





# def Move():
#     init()
#     if keyboard.is_pressed('down arrow + 1'):
#         print("Moving Backward Fast....")
#         GPIO.output(15, 1)
#         GPIO.output(16, 0)
#         GPIO.output(18, 1)
#         GPIO.output(19, 1)
# #         cv2.waitKey(50)
#         GPIO.cleanup()
#     elif keyboard.is_pressed('down arrow'):
#         print("Moving Backward Slow....")
#         GPIO.output(15, 1)
#         GPIO.output(16, 0)
#         GPIO.output(18, 0)
#         GPIO.output(19, 0)
# #         cv2.waitKey(50)
#         GPIO.cleanup()
#     elif keyboard.is_pressed('up arrow + 1'):
#         print("Moving Forward Fast....")
#         GPIO.output(15, 0)
#         GPIO.output(16, 1)
#         GPIO.output(18, 1)
#         GPIO.output(19, 1)
# #         cv2.waitKey(50)
#         GPIO.cleanup()
#     elif keyboard.is_pressed('up arrow'):
#         print("Moving Forward Slow....")
#         GPIO.output(15, 0)
#         GPIO.output(16, 1)
#         GPIO.output(18, 0)
#         GPIO.output(19, 0)
# #         cv2.waitKey(50)
#         GPIO.cleanup()
#         
#         
#         
#         
#         
# 
#     elif keyboard.is_pressed('s'):
#         print("STOP Moving....")
#         GPIO.output(35 , 1)
#         GPIO.output(36 , 1)
#         GPIO.output(15 , 1)
#         GPIO.output(16 , 1)
#         GPIO.output(18 , 1)
#         GPIO.output(19 , 1)
#


        
#     elif ActionTitle == 'B':
#         GPIO.output(15, 0)
#         GPIO.output(16, 1)
#         GPIO.output(18, 0)
#         GPIO.output(19, 0)  
#         cv2.waitKey(50)
#         GPIO.cleanup()
#     elif ActionTitle == 'F1':
#         GPIO.output(15, 0)
#         GPIO.output(16, 1)
#         GPIO.output(18, 0)
#         GPIO.output(19, 0)  
#         cv2.waitKey(50)
#         GPIO.cleanup()  
#     elif ActionTitle == 'F2':
#         GPIO.output(15, 0)
#         GPIO.output(16, 1)
#         GPIO.output(18, 1)
#         GPIO.output(19, 1)
#         cv2.waitKey(50)
#         GPIO.cleanup()
#     elif ActionTitle == 's':
#         GPIO.output(15, 0)
#         GPIO.output(16, 0)
#         GPIO.output(18, 0)
#         GPIO.output(19, 0)
#         cv2.waitKey(50)



def m_back(n):
    print('Move_back')
    init()
    n=str(abs(round(n)))
    n=int(n)
    for i in range(n):
      
        init()
        
        GPIO.setmode(GPIO.BOARD)		#set pin numbering system
        GPIO.setwarnings(False)

        GPIO.setup(15,GPIO.OUT)
        GPIO.setup(16,GPIO.OUT)
        GPIO.setup(18,GPIO.OUT)
        GPIO.setup(19,GPIO.OUT)
        GPIO.output(15, 1)
        GPIO.output(16, 0)
        GPIO.output(18, 1)
        GPIO.output(19, 1)
        


        cv2.waitKey(50)
        init()
        GPIO.setup(15,GPIO.OUT)
        GPIO.setup(16,GPIO.OUT)
        GPIO.setup(18,GPIO.OUT)
        GPIO.setup(19,GPIO.OUT)
        GPIO.output(15, 1)
        GPIO.output(16, 0)
        GPIO.output(18, 1)
        GPIO.output(19, 1)
 
    
def m_left(n):
    print('Move_Left')
    init()
    n=str(abs(round(n)))
    n=int(n)
    for i in range(n):
      
        init()
        
        GPIO.setmode(GPIO.BOARD)		#set pin numbering system
        GPIO.setwarnings(False)

        GPIO.setup(35,GPIO.OUT)
        GPIO.setup(36,GPIO.OUT)
        GPIO.output(35,0)
        GPIO.output(36,1)
        cv2.waitKey(50)
        init()
        GPIO.setup(35,GPIO.OUT)
        GPIO.setup(36,GPIO.OUT)
        GPIO.output(35,1)
        GPIO.output(36,1)
 
    
    
def m_right(n):
    n=str(abs(round(n)))
    n=int(n)
    print('Move_Right')
    init()    
    for i in range(n):
        init()
        GPIO.setmode(GPIO.BOARD)		#set pin numbering system
        GPIO.setwarnings(False)
        
        GPIO.setup(35,GPIO.OUT)
        GPIO.setup(36,GPIO.OUT)
        GPIO.output(35,1)
        GPIO.output(36,0)
        cv2.waitKey(50)
        init()
        GPIO.setup(35,GPIO.OUT)
        GPIO.setup(36,GPIO.OUT)
        GPIO.output(35,1)
        GPIO.output(36,1)
#     GPIO.setup(35,GPIO.OUT)
#     GPIO.setup(36,GPIO.OUT)
#     GPIO.output(35,1)
#     GPIO.output(36,1) 


def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)


def st_reset(scale):
    middle = int(scale/2)
    m_left(20)
    cv2.waitKey(50)
    m_right(middle)
    print('middle is ' + str(middle))
    return(middle)

def st_fun(steering,old_st):


    st = steering*steeringSen
    new=translate(st, -1, 1, 0, scale)
    
    if new > scale : new = scale
    
    if new < 0 : new=0
#     print('st' + str(st))
#     print('new' + str(new))
#     print('old_st' + str(old_st))
    
    s = abs(new - old_st)
    s = int(s)
    if s > 1:
        if new > old_st:
            m_right(s)
            old_st = old_st + s
            print('right' + str(s))
        if new < old_st:
            m_left(s)
            old_st = old_st - s
            print('left -'+ str(s))
    print("predection is "+ str(st))
    print("steering angle is "+ str(old_st))
    return(old_st)




    
#########################################################      
        
n = st_reset(scale)
old_st = n      
        

while not keyboard.is_pressed('x+z'):
    
            
    # motor move Function
    init()
    if keyboard.is_pressed('down arrow + 1'):
        print("Moving Backward Fast....")
        GPIO.output(15, 1)
        GPIO.output(16, 0)
        GPIO.output(18, 0)
        GPIO.output(19, 0)
#         cv2.waitKey(50)
        GPIO.cleanup()
    elif keyboard.is_pressed('down arrow'):
        print("Moving Backward Slow....")
        GPIO.output(15, 1)
        GPIO.output(16, 0)
        GPIO.output(18, 1)
        GPIO.output(19, 1)
#         cv2.waitKey(50)
        GPIO.cleanup()
    elif keyboard.is_pressed('up arrow + 1'):
        print("Moving Forward Fast....")
        GPIO.output(15, 0)
        GPIO.output(16, 1)
        GPIO.output(18, 0)
        GPIO.output(19, 0)
#         cv2.waitKey(50)
        GPIO.cleanup()
    elif keyboard.is_pressed('up arrow'):
        print("Moving Forward Slow....")
        GPIO.output(15, 0)
        GPIO.output(16, 1)
        GPIO.output(18, 1)
        GPIO.output(19, 1)
#         cv2.waitKey(50)
        GPIO.cleanup()
    elif keyboard.is_pressed('right arrow'):
        print("Moving right....")
        m_right(1)
    elif keyboard.is_pressed('left arrow'):
        print("Moving left....")
        m_left(1)

    elif keyboard.is_pressed('0'):
        print('reset angle')
        GPIO.output(15, 1)
        GPIO.output(16, 1)
        n = st_reset(scale)
        old_st = n
    elif keyboard.is_pressed('s'):
#             else:
        print("STOP Moving....")
        GPIO.output(35 , 1)
        GPIO.output(36 , 1)
        GPIO.output(15 , 1)
        GPIO.output(16 , 1)
        GPIO.output(18 , 1)
        GPIO.output(19 , 1)

    if keyboard.is_pressed('d'):
        print("Enter self drive mode")
        while not keyboard.is_pressed('x'):
            altra()
            
            # Moving Forward Low Speed Level

            if keyboard.is_pressed('down arrow + 1'):
                print("Moving Backward Fast....")
                GPIO.output(15, 1)
                GPIO.output(16, 0)
                GPIO.output(18, 0)
                GPIO.output(19, 0)
        #         cv2.waitKey(50)
                GPIO.cleanup()
            elif keyboard.is_pressed('down arrow'):
                print("Moving Backward Slow....")
                GPIO.output(15, 1)
                GPIO.output(16, 0)
                GPIO.output(18, 1)
                GPIO.output(19, 1)
        #         cv2.waitKey(50)
                GPIO.cleanup()
            elif keyboard.is_pressed('up arrow + 1'):
                print("Moving Forward Fast....")
                GPIO.output(15, 0)
                GPIO.output(16, 1)
                GPIO.output(18, 0)
                GPIO.output(19, 0)
        #         cv2.waitKey(50)
                GPIO.cleanup()
            elif keyboard.is_pressed('up arrow'):
                print("Moving Forward Slow....")
                GPIO.output(15, 0)
                GPIO.output(16, 1)
                GPIO.output(18, 1)
                GPIO.output(19, 1)
                
            
            elif keyboard.is_pressed('right arrow'):
                print("Moving right....")
                m_right(5)
            elif keyboard.is_pressed('left arrow'):
                print("Moving left....")
                m_left(5)

            elif keyboard.is_pressed('s'):
        #             else:
                print("STOP Moving....")
                GPIO.output(35 , 1)
                GPIO.output(36 , 1)
                GPIO.output(15 , 1)
                GPIO.output(16 , 1)
                GPIO.output(18 , 1)
                GPIO.output(19 , 1)
                
                
            elif keyboard.is_pressed('0'):
                print('reset angle')
                GPIO.output(15, 1)
                GPIO.output(16, 1)
                n = st_reset(scale)
                old_st = n
#                 GPIO.output(15, 0)
#                 GPIO.output(16, 1)


            else:
                print("Moving Forward Slow....")
                GPIO.output(15, 0)
                GPIO.output(16, 1)
                GPIO.output(18, 1)
                GPIO.output(19, 1) 
                
################################################################
            img = wM.getImg(True, size=[240, 120])
#             img = cv2.imread(image_path)

            size=[240, 120]
            img = cv2.resize(img,(size[0],size[1]))
            
            
            cv2.imshow('IMG',img)
################################################################
            
            img = cv2.resize(img, (240, 120))
            img = np.asarray(img)
            img = preProcess(img)
            img = np.array([img])
            steering = float(predict(img,input_details , output_details))
            # print(old_st)
            # motor steering Function(steering*steeringSen)
            old_st = st_fun(steering,old_st)
            #print(old_st)

            
            cv2.waitKey(50)
        init()
        GPIO.output(15 , 1)
        GPIO.output(16 , 1)
        print("STOP")



    elif keyboard.is_pressed('r+e'):
        record = 0
        print("Enter record mode")
        while not keyboard.is_pressed('x'):
            steering = sum(pyautogui.position(y=-683)) / 683
#             print(steering)

            # motor steering Function(steering)
            old_st = st_fun(steering,old_st)
            
#             print(old_st)
            
        if keyboard.is_pressed('r'):
            if record == 0: print('Recording Started ...')
            record += 1
            sleep(0.300)
        if record == 1:
            img = wM.getImg(True, size=[240, 120])
            dcM.saveData(img, steering)
        elif record == 2:
            dcM.saveLog()
            record = 0
            print("STOP Record")
        
StopAllProcess()
