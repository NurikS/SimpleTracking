from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")

results = model.predict("../cars.jpg")

img = results[0].plot()
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()