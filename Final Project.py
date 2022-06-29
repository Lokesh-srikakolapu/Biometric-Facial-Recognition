import cv2
import face_recognition
import os
from datetime import datetime
import pyttsx3


def speak(text):
    engine.say(text)
    engine.runAndWait()


def findEncodings(images):
    encodeList = []
    for img in images:
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def boxTxt(img,name = 'Unknown', color = (0,0,255)):
    cv2.rectangle(img,(x1*4,y1*4),(x2*4,y2*4),color,2)
    cv2.rectangle(img,(x1*4,y2*4-35),(x2*4,y2*4),color,cv2.FILLED)
    cv2.putText(img,name,(x1*4+6,y2*4-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)


def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = set()
        dateList = set()
        for line in myDataList:
            entry = line.split(',')
            nameList.add(entry[0])
            try:
                dateList.add(entry[1])
            except IndexError:
                pass
        present = datetime.now()
        dtString = present.strftime('%d-%m-%Y,%H:%M:%S')
        if name not in nameList:
            f.writelines(f'\n{name},{dtString}')
            attendTaken = f'You are {name}, Your Attendance taken.'
            speak(attendTaken)
        elif name in nameList and present.strftime('%d-%m-%Y') not in dateList:
            f.writelines(f'\n{name},{dtString}')
            attendTaken = f'You are {name}, Your Attendance taken.'
            speak(attendTaken)
        else:
            attended = f'You are {name}, Your Attendance is already recorded.'
            speak(attended)


path = 'pics/'
images = []
people = []
myList = os.listdir(path)
for fname in myList:
    curImg = cv2.imread(path+fname)
    people.append(fname.split('.')[0])
    images.append(curImg)
print("People Found: ",people)

try:
    encodeListKnown = findEncodings(images)
    print('Encoding Complete...')
except IndexError:
    print("Error... \nPossible issues: Provided known pictures' Face is not clear or No Face Found")
    exit()
except:
    print('Unknown Error Faced....')
    exit()

capture = cv2.VideoCapture(0)
engine = pyttsx3.init()

while True:
    success, img = capture.read()
    if success:
        scale = 0.25
        vertical = int(img.shape[0]*scale)
        horizontal = int(img.shape[1]*scale)
        imgS = cv2.resize(img,(horizontal,vertical))
        fcCurFrame = face_recognition.face_locations(imgS)
        encodeCurFrame = face_recognition.face_encodings(imgS)

        if len(fcCurFrame) == 1:
                matches = face_recognition.compare_faces(encodeListKnown,encodeCurFrame[0],0.5)
                fcDis = face_recognition.face_distance(encodeListKnown, encodeCurFrame[0])
                for i in range(len(fcDis)):
                    if min(fcDis) == fcDis[i]:
                        matchIndex = i
                        break
                y1,x2,y2,x1 = fcCurFrame[0]
                if matches[matchIndex]:
                    name = people[matchIndex]
                    boxTxt(img, name,(0,255,0))
                else:  
                    name = 'Unknown'
                    print('Unknown Face Detected...')
                    boxTxt(img)
        elif len(fcCurFrame) == 0:
            cv2.destroyAllWindows()
            continue
        else:
            print('Multiple Faces Detected...\n\tOne Person at a time...')
                
        cv2.imshow('Live Video',img)
        key = cv2.waitKey(1)
        if key == 27:
            break
        if name is not 'Unknown':
            markAttendance(name)
    else:
        print("We've faced a trouble with you Cam, Please check your Cam...")
        break

capture.release()
cv2.destroyAllWindows()