from tkinter import *
import tkinter as tk
from PIL import ImageTk, Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
from keras.applications import MobileNet
from keras.models import Sequential,Model
from keras.layers import Dense,Dropout,Activation,Flatten,GlobalAveragePooling2D
from keras.layers import Conv2D,MaxPooling2D,ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
# MobileNet is designed to work with images of dim 224,224
import  pickle
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
root = Tk()
root.title("emotion recognition")
# Create a frame
app3 = Frame(root, bg="cornflowerblue")
app3.grid(row=0, column=0)
app = Frame(root, bg="cornflowerblue")
app.grid(row=0,column=1)
app2 = Frame(root,bg="cornflowerblue")
app2.grid(row=1,column=1)
# Create a label in the frame
lmain = Label(app)
lmain.grid(row=0,column=8)
frame2=cv2.imread("img_app.jpg")
frame2=cv2.resize(frame2,(500,500))
img = Image.fromarray(frame2)
imgtk = ImageTk.PhotoImage(image=img)
lmain.imgtk = imgtk
lmain.configure(image=imgtk)
# Create a label in the frame
p0,p1,p2,p3,p4,p5,p6 ,pfinal= StringVar(),StringVar(),StringVar(),StringVar(),StringVar(),StringVar(),StringVar(),StringVar()
lmain2 = Label(app3, textvariable=p0,height=3,bg="cornflowerblue",font=20)
lmain2.grid()
lmain3 = Label(app3, textvariable=p1,height=3,bg="cornflowerblue",font=20)
lmain3.grid()
lmain4 = Label(app3, textvariable=p2,height=3,bg="cornflowerblue",font=20)
lmain4.grid()
lmain5 = Label(app3, textvariable=p3,height=3,bg="cornflowerblue",font=20)
lmain5.grid()
lmain6 = Label(app3, textvariable=p4,height=3,bg="cornflowerblue",font=20)
lmain6.grid()
lmain7 = Label(app3, textvariable=p5,height=3,bg="cornflowerblue",font=20)
lmain7.grid()
lmain8 = Label(app3, textvariable=p6,height=3,bg="cornflowerblue",font=20)
lmain8.grid()
lmainfinal = Label(app3, textvariable=pfinal,height=3,bg="lime",font=30)
lmainfinal.grid()
from pandas import DataFrame
data1 = {'emotion': ['Angry', 'Disgust', 'fear', 'Happy',
                     'Neutral', 'Sad', 'Surprise'],
         'pourcentage':[45, 42, 52,
              49, 43,49, 47]
         }
df1 = DataFrame(data1, columns=['emotion', 'pourcentage'])
comp=0
def tracer_figure(data,c):
    df1 = DataFrame(data, columns=['emotion', 'pourcentage'])
    print(df1)
    #df1.at[2,"pourcentage"]=1
    #plt.bar(data1['Country'],data1['GDP_Per_Capita'], color=['black', 'red', 'green', 'blue', 'cyan'])
    #figure1.colorbar(data1['Country'],data1['GDP_Per_Capita'], color=['black', 'red', 'green', 'blue', 'cyan'])
    my_colors = 'rgbkymc'
    figure1 = plt.Figure(figsize=(4,3), dpi=100)
    ax1 = figure1.add_subplot(111)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    #ax1.bar(color=['black', 'red', 'green', 'blue', 'cyan'])
    color2 = list('rgbkymc')
    bar1 = FigureCanvasTkAgg(figure1, app3)
    bar1.get_tk_widget().grid()
    df1 = df1[['emotion','pourcentage']].groupby('emotion').sum()
    df1.plot(kind='barh', legend=True, ax=ax1,color="r")
    ax1.get_children()[2].set_color('b')
    ax1.set_title('Emotion recognition')
    ax1=None
    df1=None
    if c != 0:
        figure1.close()
    #bar1.bar1.get_tk_widget().destroy()
def tracer_figure2(data):
    df1 = DataFrame(data, columns=['emotion', 'pourcentage'])
    print(df1)
    #df1.at[2,"pourcentage"]=1
    #plt.bar(data1['Country'],data1['GDP_Per_Capita'], color=['black', 'red', 'green', 'blue', 'cyan'])
    #figure1.colorbar(data1['Country'],data1['GDP_Per_Capita'], color=['black', 'red', 'green', 'blue', 'cyan'])
    #my_colors = 'rgbkymc'
    figure1 = plt.Figure(figsize=(4,3), dpi=100)
    ax1 = figure1.add_subplot(111)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    #ax1.bar(color=['black', 'red', 'green', 'blue', 'cyan'])
    #color2 = list('rgbkymc')
    bar1 = FigureCanvasTkAgg(figure1, app3)
    bar1.get_tk_widget().grid()
    df1 = df1[['emotion','pourcentage']].groupby('emotion').sum()
    df1.plot(kind='barh', legend=True, ax=ax1,color="r")
    ax1.get_children()[2].set_color('b')
    ax1.set_title('Emotion recognition')
    #bar1.bar1.get_tk_widget().destroy()

#import bar_chart_race as bcr
#bcr.bar_chart_race(df=df1,filename=None,figsize=(3.5,3),title='test')
"""df1 = DataFrame(data1, columns=['emotion', 'pourcentage'])
print(df1)
df1.at[2,"pourcentage"]=1
#plt.bar(data1['Country'],data1['GDP_Per_Capita'], color=['black', 'red', 'green', 'blue', 'cyan'])
#figure1.colorbar(data1['Country'],data1['GDP_Per_Capita'], color=['black', 'red', 'green', 'blue', 'cyan'])
my_colors = 'rgbkymc'
figure1 = plt.Figure(figsize=(4,3), dpi=100)
ax1 = figure1.add_subplot(111)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
#ax1.bar(color=['black', 'red', 'green', 'blue', 'cyan'])
color2 = list('rgbkymc')
bar1 = FigureCanvasTkAgg(figure1, app3)
bar1.get_tk_widget().grid()
df1 = df1[['emotion','pourcentage']].groupby('emotion').sum()
df1.plot(kind='barh', legend=True, ax=ax1,color="r")
ax1.get_children()[2].set_color('b')
ax1.set_title('Emotion recognition')
ax1=None
"""
# Create buttom
# Capture from camera
cap = cv2.VideoCapture(0)
# function for video streaming
face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
classifier = load_model('my_model/model-top1.h5')
class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
class_labels2 = ['Angry', 'Disgust', 'fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
#tracer_figure()
#tracer_figure(data1)
#tracer_figure(data1)
def video_stream():
    _, frame = cap.read()
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(cv2image,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = cv2image[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)
        # make a prediction on the ROI, then lookup the class
            preds = classifier.predict(roi)
            print("\nprediction = ",preds)
            label=class_labels2[preds.argmax()]
            print("pred0+",preds[0])
            label0 = class_labels2[0]
            label1 = class_labels2[1]
            label2 = class_labels2[2]
            label3 = class_labels2[3]
            label4 = class_labels2[4]
            label5 = class_labels2[5]
            label6 = class_labels2[6]
            labelfinal=class_labels2[preds.argmax()]
            print("\nprediction max = ",preds.argmax())
            print("\nlabel = ",label)
            p0.set(label0+" avec purcentage is  = "+str(round(preds[0][0]*100,2)))
            p1.set(label1+" \U0001f600 avec purcentage is  = "+str(round(preds[0][1]*100,2)))
            p2.set(label2+" avec purcentage is  = "+str(round(preds[0][2]*100,2)))
            p3.set(label3+" avec purcentage is  = "+str(round(preds[0][3]*100,2)))
            p4.set(label4+" avec purcentage is  = "+str(round(preds[0][4]*100,2)))
            p5.set(label5+" avec purcentage is  = "+str(round(preds[0][5]*100,2)))
            p6.set(label6+" avec purcentage is  = "+str(round(preds[0][6]*100,2)))
            pfinal.set(labelfinal + " avec purcentage is  = " + str(round(preds[0][preds.argmax()] * 100, 2)))
            label_position = (x,y-50)
            label_position2=(x,y)
            cv2.putText(frame,label,label_position2,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)
            """data1 = {'emotion': ['Angry', 'Disgust', 'fear', 'Happy',
                                 'Neutral', 'Sad', 'Surprise'],
                     'pourcentage': [45, 42, 52,
                                     49, 43, 49, 47]
                     }"""
            #tracer_figure(data1)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(1, video_stream)
    return preds
def start():
    video_stream()
    tracer_figure()
from tkinter import filedialog
def browseFiles():
    filename = filedialog.askopenfilename(initialdir="/",
                                          title="Select a File",
                                          filetypes=(("image files",
                                                      "*.jpg*"),
                                                     ("all files",
                                                      "*.*")))
    print('file name is '+filename)
    return filename

    # Change label contents
    #label_file_explorer.configure(text="File Opened: " + filename)
def choisir_imag():
    file=browseFiles()
    frame2 = cv2.imread(file)
    frame2 = cv2.resize(frame2, (300, 300))
    img = Image.fromarray(frame2)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
def traiter_sentiment_img():
    file = browseFiles()
    frame = cv2.imread(file)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(cv2image, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = cv2image[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            # make a prediction on the ROI, then lookup the class
            preds = classifier.predict(roi)
            print("\nprediction = ", preds)
            label = class_labels2[preds.argmax()]
            """print("pred0+",preds[0])
            label0 = class_labels2[0]
            label1 = class_labels2[1]
            label2 = class_labels2[2]
            label3 = class_labels2[3]
            label4 = class_labels2[4]
            label5 = class_labels2[5]
            label6 = class_labels2[6]"""
            print("\nprediction max = ", preds.argmax())
            print("\nlabel = ", label)
            """p0.set(label0+" avec purcentage is  = "+str(preds[0][0]*100))
            p1.set(label1+" avec purcentage is  = "+str(preds[0][1]))
            p2.set(label2+" avec purcentage is  = "+str(preds[0][2]))
            p3.set(label3+" avec purcentage is  = "+str(preds[0][3]))
            p4.set(label4+" avec purcentage is  = "+str(preds[0][4]))
            p5.set(label5+" avec purcentage is  = "+str(preds[0][5]))
            p6.set(label6+" avec purcentage is  = "+str(preds[0][6]))"""  #
            label_position = (x, y - 50)
            label_position2 = (x, y)
            cv2.putText(frame, label, label_position2, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            frame = cv2.resize(frame, (500, 500))
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    d2=preds[0]
    data1 = {'emotion': ['Angry', 'Disgust', 'fear', 'Happy',
                         'Neutral', 'Sad', 'Surprise'],
             'pourcentage': d2
             }
    comp=0
    tracer_figure(data1,comp)
    comp = comp + 1
    return None
#Create a File Explorer label
#label_file_explorer = Label(root,text="File Explorer using Tkinter",width=100, height=4,fg="blue")
button_explore = Button(app2,text="Browse Files",command=traiter_sentiment_img, width=10, height=1, bg='#0052cc', fg='#ffffff', activebackground='#0052cc', activeforeground='#aaffaa')
button_exit = Button(app2,text="Exit",command=exit, width=10, height=1, bg='#0052cc', fg='#ffffff', activebackground='#0052cc', activeforeground='#aaffaa')
redbutton = Button(app2, text="Start le stream ",command=video_stream,  width=10, height=1, bg='#0052cc', fg='#ffffff', activebackground='#0052cc', activeforeground='#aaffaa')
redbutton.grid(column=6,row=11)
#label_file_explorer.grid(column=1, row=1)
button_explore.grid(column=7,row=11)
button_exit.grid(column=8,row=11)
#video_stream()
root.mainloop()