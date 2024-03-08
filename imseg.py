import cv2
import numpy as np
import logging

logging.basicConfig(filename="LoggerIS.log", format='%(asctime)s %(message)s', filemode='w')
lg = logging.getLogger()
lg.debug("File selector window about to be created")


class bgremer:
    def __init__(self, overlay):
        self.drawing = False
        self.top_left_point, self.bottom_right_point = (-1, -1), (-1, -1)
        try:
            self.original = cv2.imread(overlay)
            # h, w ,c = self.original.shape
            # self.original = cv2.resize(self.original, (500, 500))
            self.image = self.original.copy()
            cv2.namedWindow("Apparel Select")
            cv2.setMouseCallback("Apparel Select", self.drawbbox)
        except:
            lg.warning("Check File type")
            quit()

        while True:
            cv2.imshow("Apparel Select", self.image)
            c = cv2.waitKey(1)
            if c == 27:
                break

    def drawbbox(self, click, x, y, flag, param):
        global x_pt, y_pt, drawing, top_l_pt, bot_r_pt, original

        if click == cv2.EVENT_LBUTTONDOWN:
            lg.info("BBox Lbutton clicked")

            self.drawing = True
            x_pt, y_pt = x, y

        elif click == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                lg.info("BBox mouse moved")
                top_l_pt, bot_r_pt = (x_pt, y_pt), (x, y)
                self.image[y_pt:y, x_pt:x] = 255 - self.original[y_pt:y, x_pt:x]
                cv2.rectangle(self.image, top_l_pt, bot_r_pt, (0, 255, 100), 2)


        elif click == cv2.EVENT_LBUTTONUP:
            lg.info("BBox Lbutton letoff")
            self.drawing = False
            top_l_pt, bot_r_pt = (x_pt, y_pt), (x, y)
            self.image[y_pt:y, x_pt:x] = 255 - self.image[y_pt:y, x_pt:x]
            cv2.rectangle(self.image, top_l_pt, bot_r_pt, (0, 255, 100), 2)
            bbox = (x_pt, y_pt, x - x_pt, y - y_pt)

            self.grabcut_alg(self.original, bbox)

    def grabcut_alg(self, original, bbox):
        lg.info("In grabcut method")
        segment = np.zeros(original.shape[:2], np.uint8)

        x, y, width, height = bbox
        segment[y:y + height, x:x + width] = 1

        background_mdl = np.zeros((1, 65), np.float64)
        foreground_mdl = np.zeros((1, 65), np.float64)

        cv2.grabCut(self.original, segment, bbox, background_mdl, foreground_mdl, 5,
                    cv2.GC_INIT_WITH_RECT)

        new_mask = np.where((segment == 2) | (segment == 0), 0, 1).astype('uint8')

        cut_img = self.original * new_mask[:, :, np.newaxis]

        cv2.imwrite('cut_image.png', cut_img)
        cv2.imshow('Result', cut_img)
