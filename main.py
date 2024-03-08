import cv2
import mediapipe as mp
import time
import imseg
import numpy as np
from PIL import Image, ImageFilter
import logging

logging.basicConfig(filename="LoggerM.log", format='%(asctime)s %(message)s', filemode='w')
lg2 = logging.getLogger()

lg2.info("Program Started")

frw = 1280
frh = 720
FILENAME = 'gshirt.jpg'

lg2.info("passing file name to image segmenting program")
imseg.bgremer(FILENAME)
# print("next")

lg2.info("Creating cv2 live video object")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frw)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frh)

pTime = 0

lg2.debug("Getting pose estimation models")
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()


def repwalpha():
    # replacing black shades with transparent pixels
    src = cv2.imread("cut_image.png", 1)
    tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)

    b, g, r = cv2.split(src)

    rgba = [b, g, r, alpha]

    dst = cv2.merge(rgba, 4)
    cv2.imwrite("cut_image.png", dst)
    # Writing and saving to a new image
    # cv2.imshow('frame', dst)
    cv2.waitKey(1)


def find_center(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5),
                            cv2.BORDER_DEFAULT)
    ret, thresh = cv2.threshold(blur, 200, 255,
                                cv2.THRESH_BINARY_INV)

    contours, hierarchies = cv2.findContours(
        thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    blank = np.zeros(thresh.shape[:2],
                     dtype='uint8')

    cv2.drawContours(blank, contours, -1,
                     (255, 0, 0), 1)
    for i in contours:
        M = cv2.moments(i)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.drawContours(image, [i], -1, (0, 255, 0), 2)
            cv2.circle(image, (cx, cy), 7, (0, 0, 255), -1)
    return (cx, cy)


while True:
    lg2.info("live video frame read")
    ret, frame = cap.read()
    RGBframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(RGBframe)
    # print(results.pose_landmarks)
    if (results.pose_landmarks):
        # mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        ymid = 0
        xmid = 0
        shoulderx = 0
        shouldery = 0
        hipx = 0
        hipy = 0
        for id, lm in enumerate(results.pose_landmarks.landmark):
            if id == 11:
                shoulderx = int(lm.x * w)
                shouldery = int(lm.y * h)
            elif id == 24:
                hipx = int(lm.x * w)
                hipy = int(lm.y * h)
            h, w, c = frame.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            # print(id, cx, cy)
            # print(shoulder, hip)
            xmid = int((shoulderx + hipx) // 2)
            ymid = int((shouldery + hipy) // 2)
            # cv2.circle(frame, (cx,cy), 10, (255,0,150), 2)
            if id == 32:
                lg2.info("At midpoint")
                cv2.circle(frame, (xmid, ymid), 10, (0, 255, 150), 2)
                # print("check 1")
                colconvframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # print("check 2")
                pil_frame = Image.fromarray(colconvframe)
                lg2.info("Converted cv2 object to PIL object")
                # print('about to go in')
                repwalpha()
                # print("Got back")
                overlay = Image.open("cut_image.png")
                # width, height = overlay.size

                centerx, centery = find_center(np.array(overlay))
                newim = Image.new("RGBA", pil_frame.size)
                newim.paste(overlay, ((xmid - centery), ((ymid - centerx) - 100)))
                # print("check 3")
                # print("paste done")

                # newim.putalpha(225)
                # blur = Image.new("RGBA", newim.size)
                # blur = newim.filter(ImageFilter.GaussianBlur(3))
                # newim.putalpha(blur)

                pil_frame.paste(newim, mask=newim)
                npimg = np.array(pil_frame)
                lg2.info("Frame converted back to cv2 from PIL")
                frame = cv2.cvtColor(npimg, cv2.COLOR_RGB2BGR)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(frame, str(int(fps)), (20, 80), cv2.FONT_HERSHEY_PLAIN, 3, (0, 250, 0), 3)
    cv2.imshow("Frame Cap", frame)

    c = cv2.waitKey(1)
    if c == 27:
        break