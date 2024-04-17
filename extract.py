import cv2
import sys
import os

vidcap = cv2.VideoCapture(str(sys.argv[1]))
success,image = vidcap.read()
count = 0
length = 7
calc = 0
digits = ""

if not os.path.exists(os.path.dirname(os.path.realpath(str(sys.argv[1])))+"/imgs"):
    os.makedirs(os.path.dirname(os.path.realpath(str(sys.argv[1])))+"/imgs")
 
while success:
  calc = length - len(str(count))
  digits = ""
  for i in range(calc):
    digits += "0"
    
  cv2.imwrite( os.path.dirname(os.path.realpath(str(sys.argv[1])))+"/imgs/"+digits + str(count) + ".png", image )
  #cv2.imwrite( os.path.dirname(os.path.realpath(str(sys.argv[1])))+"/imgs/"+digits + str(count) + ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1
