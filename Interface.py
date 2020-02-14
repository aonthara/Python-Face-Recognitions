import numpy
from pygame import mixer
import time
import cv2
from tkinter import *
import tkinter.messagebox
from tkinter import messagebox
import numpy as np
from joblib import load
from playsound import playsound
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump, load
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib



root=Tk()
root.geometry('520x530')
frame = Frame(root, relief=RIDGE, borderwidth=2)
frame.pack(fill=BOTH,expand=1)
root.title('Detection-Program')
frame.config(background='black')
label = Label(frame, text=" Detection-Program ",bg='white',font=('Times 36 bold'))
label.pack(side=TOP)
filename = PhotoImage(file="C:/Users/thara/Desktop/FaceReconition/img/bl.png")
background_label = Label(frame,image=filename)
background_label.pack(side=TOP)

def hel():
   help(cv2)

def Contri():
   tkinter.messagebox.showinfo("ผู้ออกแบบ ","\n 1. อ้น \n 2. เฟริสท์ \n")


def anotherWin():
   tkinter.messagebox.showinfo("อ้างอิง",'Face Recognition\n Made Using\n-OpenCV\n-Numpy\n-Tkinter\n-Pandas\n-Joblib\n-Matplotlib\n-Sklearn\n-Pygame\n-Time\n-Dlib\n-Imutils\n-Python 3')


def sa():
   OUTPUT_PATH = 'datasets/faces'
   SIZE = (64,64)
   MAX_CAPTURE = 100
   detector = cv2.CascadeClassifier('C:\\Users\\thara\\Desktop\\FaceReconition\\model-haar\\haarcascade_frontalface_default.xml')

   font = cv2.FONT_HERSHEY_SIMPLEX
   color = (0,255,0)

   label = input('กรุณากรอกชื่อ : ')
   output_path = Path(OUTPUT_PATH)
   if not output_path.exists():
      output_path.mkdir()
   output_face_path = Path(OUTPUT_PATH + '/' + label)
   if not output_face_path.exists():
      output_face_path.mkdir()    

   count = 0
   capture = cv2.VideoCapture(0)
   while count < MAX_CAPTURE:
      ret, frame = capture.read()
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      faces = detector.detectMultiScale(gray, 1.2, 4)
      if len(faces) > 0:
         for (x, y, w, h) in faces:
               if w >= SIZE[0]:
                  face = frame[y:y+h, x:x+w]
                  output_name = OUTPUT_PATH +  '/' + label + '/img' + str(count) + '.jpg'
                  face_cropped = frame[y:y+h, x:x+w]
                  face_resized = cv2.resize(face_cropped, SIZE, interpolation=cv2.INTER_LINEAR)
                  cv2.imwrite(output_name, face_resized)
                  cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
                  cv2.putText(frame, ' all = ' + str(count) + ' / ' + str(MAX_CAPTURE), (x,y-10), font, 0.6, color, thickness=2)
                  count += 1
      cv2.imshow('Face-Auto-Save', frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
         break
   capture.release()
   cv2.destroyAllWindows()
    
def fa():
   HAAR_MODEL = 'C:/Users/thara/Desktop/FaceReconition/model-haar/haarcascade_frontalface_default.xml'
   SVM_MODEL = 'C:/Users/thara/Desktop/FaceReconition/model-svm/faces.lib'

   font = cv2.FONT_HERSHEY_SIMPLEX
   color_deful = (255,100,0)
   color_unknown = (0,0,255)
   threshold = 0.8

   detector = cv2.CascadeClassifier(HAAR_MODEL)
   classifier = load(SVM_MODEL)

   capture = cv2.VideoCapture(0)
   while True:
      if capture is None:
         break
      ret, frame = capture.read()
      image = frame.copy()
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      faces = detector.detectMultiScale(gray, 1.3, 5)         

      for (x, y, w, h) in faces:
         testset = []
         face = gray[y:y+h, x:x+w]
         face_resized = cv2.resize(face,(64,64),interpolation=cv2.INTER_LINEAR)
         testset.append(np.ravel(face_resized))

         pred = str(classifier.predict(testset))
         prob = classifier.predict_proba(testset)        
         
        
         max_prob = max(prob[0])
        
         if max_prob >= threshold:
               text = ''.join(pred + ' (' + '{0:.2g}'.format(max_prob * 100) + '%)')
               color = color_deful
               if pred.find('deful') != -1:
                   color = color_deful        
               cv2.putText(image, text, (x,y-10), font, 0.6, color, thickness=2)
         else:
            color = color_unknown
            #playsound('D:\\FaceReconition\\Sound\\digital.mp3')
            time.sleep(0.1)
            
         cv2.rectangle(image, (x,y), (x+w,y+h), color, 2)
      cv2.imshow('Face-Reconition', image)
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break
   capture.release()
   cv2.destroyAllWindows()

menu = Menu(root)
root.config(menu=menu)

subm1 = Menu(menu)
menu.add_cascade(label="เมนู",menu=subm1)
subm1.add_command(label="ออก",command=exit)

subm2 = Menu(menu)
menu.add_cascade(label="เกี่ยวกับ",menu=subm2)
subm2.add_command(label="ไลบรารี่",command=anotherWin)
subm2.add_command(label="ผู้ออกแบบ",command=Contri)


def ta():
   HAAR_MODEL = './model-haar/haarcascade_frontalface_default.xml'
   INPUT_IMAGE_PATH = './datasets/faces'
   OUTPUT_CSV_FILE = './datasets/faces.csv'
   PROCESSED_IMAGE_PATH = './datasets/processed'
   PROCESSED_CSV_FILE = './datasets/processed.csv'
   DETECTED_FACE_PATH = './datasets/cropped'
   DETECTED_CSV_FILE = './datasets/cropped.csv'
   OUTPUT_MODEL_NAME = './model-svm/faces.lib'

   def create_csv(dataset_path, output_csv):
      root_dir = Path(dataset_path)
      items = root_dir.iterdir()

      filenames = []
      labels = []
      print('อ่านรูปภาพจากไฟล์ ... ')
      for item in items:
         if item.is_dir():
            for file in item.iterdir():
                  if file.is_file():
                     print(str(file))
                     filenames.append(file)
                     labels.append(item.name)
      raw_data = {'filename': filenames, 'label': labels}
      df = pd.DataFrame(raw_data, columns=['filename','label'])
      df.to_csv(output_csv)
      print(len(filenames), 'อ่านไฟล์รูปภาพ')
      # input("Press [ENTER] key to continue...")

   def resize(image, width=None, height=None):
      (h, w) = image.shape[:2]
      if width is None:
         r = height/float(h)
         dim = (int(w*r), height)
      else:
         r = width/float(w)
         dim = (width, int(h*r))
      return cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)

   def process_image(input_csv, output_csv, output_path_name):
      dataset = pd.read_csv(input_csv, sep=',')
      ids = dataset.values[:,0]
      names = dataset.values[:,1]
      labels = dataset.values[:,2]

      output_path = Path(output_path_name)
      if not output_path.exists():
         output_path.mkdir()

      filenames = []
      print('ประมวลผลภาพล่วงหน้า ... ')
      for item in names:
         input_path = Path(item)
         if input_path.is_file():
               output_name = output_path_name + '/image' + str(ids[len(filenames)]) + input_path.suffix
               print(input_path, '->', output_name)
               image = cv2.imread(str(input_path))
               image = resize(image, width=64)
               cv2.imwrite(output_name, image)
               filenames.append(output_name)
      prc_data = {'filename': filenames, 'label': labels}
      df = pd.DataFrame(prc_data, columns=['filename', 'label'])
      df.to_csv(output_csv)
      print(len(filenames), 'ประมวลผลไฟล์ภาพแล้ว')
      # input("Press [ENTER] key to continue...")

   def detect_face(input_csv, output_csv, output_path_name):
      dataset = pd.read_csv(input_csv, sep=',')
      ids = dataset.values[:,0]
      names = dataset.values[:,1]
      labels = dataset.values[:,2]

      output_path = Path(output_path_name)
      if not output_path.exists():
         output_path.mkdir()

      clf = cv2.CascadeClassifier(HAAR_MODEL)
      face_filenames = []
      face_labels = []
      count = 0
      face_count = 0
      print('ตรวจจับใบหน้า ... ')
      for item in names:
         image = cv2.imread(item)
         face_label = labels[count]

         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
         faces = clf.detectMultiScale(gray, 1.3, 5)
         for (x, y, w, h) in faces:
            cropped = image[y:y+h, x:x+w]
            output_file = output_path_name + '/face' + str(len(face_filenames)) + '.jpg'
            cv2.imwrite(output_file, cropped)

            face_filenames.append(output_file)
            face_labels.append(face_label)
         print(item, '->', len(faces), ' face(s) detected')
         face_count += len(faces)
         count += 1
      crp_data = {'filename': face_filenames, 'label': face_labels}
      df = pd.DataFrame(crp_data, columns=['filename', 'label'])
      df.to_csv(output_csv)
      print('Total of', face_count, 'face(s) detected')
      # input("Press [ENTER] key to continue...")

   def train_model(train_csv, output_model_name):
      dataset = pd.read_csv(train_csv, sep=',')
      ids = dataset.values[:,0]
      names = dataset.values[:,1]
      labels = dataset.values[:,2]

      images = []
      print('อบรมโมเดลให้รู้จำใบหน้า ...')
      for item in names:
         image = cv2.imread(str(item), 0)
         resized = cv2.resize(image, (64,64), interpolation=cv2.INTER_LINEAR)
         images.append(np.ravel(resized))

      clf = SVC(kernel='linear', probability=True)
      clf.fit(images, labels)
      dump(clf, output_model_name)
      print('ส้รางโมเดลใน...', output_model_name)
      # input("Press [ENTER] key to continue...")

   def validate_model(validate_csv, model_name):
      dataset = pd.read_csv(validate_csv, sep=',')
      ids = dataset.values[:,0]
      names = dataset.values[:,1]
      labels = dataset.values[:,2]

      images = []
      print('ตรวจสอบรูปแบบการรู้จำใบหน้า ...')
      for item in names:
         image = cv2.imread(str(item), 0)
         resized = cv2.resize(image, (64,64), interpolation=cv2.INTER_LINEAR)
         images.append(np.ravel(resized))

      clf = load(model_name)
      y_p = cross_val_predict(clf, images, labels, cv=3)
      print('คะแนนความละเอีด:', '{0:.4g}'.format(accuracy_score(labels,y_p) * 100), '%')
      print('ความเข้ากับของเมทริกซ์:')
      print(confusion_matrix(labels,y_p))
      print('รายงานการจำแนกประเภท:')
      print(classification_report(labels,y_p))

   create_csv(INPUT_IMAGE_PATH, OUTPUT_CSV_FILE)
   process_image(OUTPUT_CSV_FILE, PROCESSED_CSV_FILE, PROCESSED_IMAGE_PATH)
   train_model(PROCESSED_CSV_FILE, OUTPUT_MODEL_NAME)
   validate_model(PROCESSED_CSV_FILE, OUTPUT_MODEL_NAME)


def dd():
   def eye_aspect_ratio(eye):
	   A = distance.euclidean(eye[1], eye[5])
	   B = distance.euclidean(eye[2], eye[4])
	   C = distance.euclidean(eye[0], eye[3])
	   ear = (A + B) / (2.0 * C)
	   return ear
	
   thresh = 0.25
   frame_check = 6
   detect = dlib.get_frontal_face_detector()
   predict = dlib.shape_predictor("C:\\Users\\thara\\Desktop\\FaceReconition\\model-haar\\shape_predictor_68_face_landmarks.dat")

   (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
   (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
   
   cap=cv2.VideoCapture(0)
   flag=0
   while True:
	   ret, frame=cap.read()
	   frame = imutils.resize(frame, width=720)
	   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	   subjects = detect(gray, 0)
	   for subject in subjects:
		   shape = predict(gray, subject)
		   shape = face_utils.shape_to_np(shape)
		   leftEye = shape[lStart:lEnd]
		   rightEye = shape[rStart:rEnd]
		   leftEAR = eye_aspect_ratio(leftEye)
		   rightEAR = eye_aspect_ratio(rightEye)
		   ear = (leftEAR + rightEAR) / 2.0
		   leftEyeHull = cv2.convexHull(leftEye)
		   rightEyeHull = cv2.convexHull(rightEye)
		   cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		   cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		   if ear < thresh:
			   flag += 1
			   print (flag)
			   if flag >= frame_check:
				   cv2.putText(frame, playsound('D:\\FaceReconition\\Sound\\nasty.mp3'), (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		   else:
			   flag = 0
	   cv2.imshow("Drowsiness-Detection", frame)
	   key = cv2.waitKey(1) & 0xFF
	   if key ==ord("q"):
		   break
   cv2.destroyAllWindows()
   cap.release()


def exitt():
   exit()



but1=Button(frame,padx=5,pady=5,width=30,bg='white',fg='black',relief=GROOVE,command=sa,text='ระบบบันทึกใบหน้า',font=('helvetica 15 bold'))
but1.place(x=55,y=120)

but2=Button(frame,padx=5,pady=5,width=30,bg='white',fg='black',relief=GROOVE,command=ta,text='ระบบรู้จำใบหน้า',font=('helvetica 15 bold'))
but2.place(x=55,y=190)

but3=Button(frame,padx=5,pady=5,width=30,bg='white',fg='black',relief=GROOVE,command=fa,text='ระบบจดจำใบหน้าและระบุตัวตน',font=('helvetica 15 bold'))
but3.place(x=55,y=260)

but3=Button(frame,padx=5,pady=5,width=30,bg='white',fg='black',relief=GROOVE,command=dd,text='ระบบตรวจจับการหลับตา',font=('helvetica 15 bold'))
but3.place(x=55,y=330)

but5=Button(frame,padx=5,pady=5,width=10,bg='white',fg='black',relief=GROOVE,text='ออก',command=exitt,font=('helvetica 12 bold'))
but5.place(x=210,y=420)


root.mainloop()


