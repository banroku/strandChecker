"""save images every 0.5 sec in input movie
"""

import cv2

INPUT_TITLE = 'movie08'
INPUT_MOVIE = 'movie/' + INPUT_TITLE + '.mp4'
OUTPUT_TITLE = INPUT_TITLE
OUTPUT_SIZE = (224, 224)
INTERVAL = 30  # in frame
cap = cv2.VideoCapture(INPUT_MOVIE)

rep, frame = cap.read()
INPUT_WIDTH = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
INPUT_HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

print('input size = ', INPUT_WIDTH, INPUT_HEIGHT)

i = 1
while rep is True:
    if i % INTERVAL == 0:
        OUTPUT_NUM = int(i/INTERVAL)
        OUTPUT_FILE = 'image/original/' + OUTPUT_TITLE + \
            '_' + '{:0>4}'.format(OUTPUT_NUM) + '.bmp'
#        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#        frame_trimed = frame_gray[290:1010, :]
        frame_trimed = frame[290:1010, :]
        frame_resized = cv2.resize(frame_trimed, OUTPUT_SIZE)
        cv2.imwrite(OUTPUT_FILE, frame_resized)
    rep, frame = cap.read()
    i += 1

cap.release()
