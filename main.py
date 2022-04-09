from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.factory import Factory
from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup
from kivy.uix.image import Image
import os
from kivy.lang import Builder
import matplotlib.pyplot as plt
from imutils import contours, perspective
import numpy as np
import cv2

KV = """
Root:
    img: img
    textT: textT

    canvas.before:
        Color:
            rgba: 1, 1, 1, 1
        Rectangle:
            pos: self.pos
            size: self.size

    BoxLayout:
        orientation: 'vertical'
        Image:
            id: img
            size_hint: (.8, .8)
            pos_hint: {'center_x':0.5, 'center_y': 0}

        TextInput:
            id: textT
            size_hint_y: .4


        BoxLayout:
            size_hint_y: .2
            height: 60
            Button:
                text: 'Load'
                on_release: root.show_load()


<LoadDialog>:
    BoxLayout:
        size: root.size
        pos: root.pos
        orientation: "vertical"
        FileChooserListView:
            id: filechooser

        BoxLayout:
            size_hint_y: None
            height: 30
            Button:
                text: "Cancel"
                on_release: root.cancel()

            Button:
                text: "Load"
                on_release: root.load(filechooser.path, filechooser.selection)
"""


def SquareCalculate(pixelReferenceCount, referenceSquare, pixelObjectCount):
    pixelSquare = referenceSquare / pixelReferenceCount
    return pixelObjectCount * pixelSquare


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def makeBoxContours(c, orig, square="Square not found"):
    box = cv2.minAreaRect(c)
    box = cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    box = perspective.order_points(box)

    cv2.drawContours(orig, c, -1, (0, 0, 255), 4) # draw an original contour

    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

    for (x, y) in box:
        cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

    # the midpoint between bottom-left and bottom-right coordinates
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)
    # compute the midpoint between the top-left and top-right points,
    # followed by the midpoint between the top-right and bottom-right
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)
    # draw the midpoints on the image
    cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
    # draw lines between the midpoints
    cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)

    cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

    # draw the object sizes on the image
    cv2.putText(orig, str(square), (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 5)
    # show the output image


def findContours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    edged = cv2.Canny(gray, 40, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # find contours in the edge map
    cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

    # sort the contours
    (cnts, _) = contours.sort_contours(cnts, method="left-to-right")
    return cnts

def mainCalcSquare(img, referenceSquare=0.0003431):
    image = cv2.imread(img)
 
    cnts = findContours(image)
    sCnts = []
    squares = []

    # loop over the contours
    for c in cnts:
        square = cv2.contourArea(c)
        # continue all noisy contours
        if square < 1000:
            continue

        sCnts.append(square)

        if len(sCnts) == 1: # check if "c" is the reference (first) contour
            makeBoxContours(c, image, referenceSquare)
        else:
            makeBoxContours(c, image, SquareCalculate(sCnts[0], referenceSquare, sCnts[-1]))
            squares.append(SquareCalculate(sCnts[0], referenceSquare, sCnts[-1]))

    cv2.imwrite("output-Image.jpg", image)
    return squares


class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


class Root(FloatLayout):
    loadfile = ObjectProperty(None)
    savefile = ObjectProperty(None)
    lbl_path = ObjectProperty(None)
    img = ObjectProperty(None)
    textT = ObjectProperty(None)
    picture_path = ""

    def dismiss_popup(self):
        self._popup.dismiss()

    def show_load(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content, size_hint=(0.9, 0.9))
        self._popup.open()

    def load(self, path, filename):
        with open(os.path.join(path, filename[0])) as img:
            squares = mainCalcSquare(img.name)
            self.img.source = os.path.abspath("output-Image.jpg")
            print(squares)
            self.textT.text = str(squares)

        self.dismiss_popup()

class MainApp(App):
    def build(self):
        return Builder.load_string(KV)


Factory.register('Root', cls=Root)
Factory.register('LoadDialog', cls=LoadDialog)

if __name__ == '__main__':
    MainApp().run()
