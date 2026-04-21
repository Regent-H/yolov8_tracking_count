import cv2
import matplotlib.pyplot as plt

video = cv2.VideoCapture("./Videos/traffic.mp4")

num = 0
save_step = 50
while True:
    ret ,frame = video.read()
    if not ret:
        break
    num = num+1
    if num % save_step == 35:
        frame = cv2.resize(frame,(1280,720))
        cv2.imwrite('./demo3/' + str(num) + '.jpg', frame)