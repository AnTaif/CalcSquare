import matplotlib.pyplot as plt
from imutils import contours, perspective
import numpy as np
import cv2

def SquareCalculate(pixelReferenceCount, referenceSquare, pixelObjectCount):
    pixelSquare = referenceSquare / pixelReferenceCount
    return pixelObjectCount * pixelSquare


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def makeBoxContours(c, orig, square=0):
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
    return squares, image

if __name__ == "__main__":
    img = "input-image.jpg"

    square, image = mainCalcSquare(img)

    print(square)
    cv2.imwrite("output-image.jpg", image)
    plt.imshow(image)
    plt.show()
    
