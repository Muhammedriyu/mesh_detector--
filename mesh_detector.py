import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot

cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)
plotY = LivePlot(640, 360, [20, 50])

idlist = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
ratioList = []



while True:

    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    suceess, img = cap.read()
    img, faces = detector.findFaceMesh(img, draw= False)

    if faces:
        face = faces[0]
        for id in idlist:
            cv2.circle(img, face[id], 5, (255, 0, 255), cv2.FILLED)

        leftUp = face[159]
        leftDown = face[23]
        leftleft = face[130]
        leftRight = face[243]
        lenghtVer,_ = detector.findDistance(leftUp, leftDown)
        lenghtHor,_ = detector.findDistance(leftleft, leftRight)

        cv2.line(img, leftUp, leftDown, (0, 200, 0), 3)
        cv2.line(img, leftleft, leftRight, (0, 200, 0), 3)

        ratio = int((lenghtVer/lenghtHor) * 100)
        ratioList.append(ratio)
        if len(ratioList)>10:
            ratioList.pop(0) 
        ratioAvg = sum(ratioList)/len(ratioList)
        imgPlot = plotY.update(ratioAvg)
        img = cv2.resize(img, (640, 360))
        imgStack = cvzone.stackImages([img, imgPlot], 2, 1)
    else:
        img = cv2.resize(img, (640, 360))
        imgStack = cvzone.stackImages([img, img], 2, 1)


    cv2.imshow("Image", imgStack)
    cv2.waitKey(25)
