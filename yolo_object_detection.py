import cv2
import numpy as np

__win_name = "object detection"
__network = cv2.dnn.readNet("../yolov3.weights", "yolov3.cfg")
with open("coco_classes.txt", "r") as file:
    __classes = [line.strip() for line in file.readlines()]
print(__classes)
# with open("coco_classes.txt", "r") as f:
#     __classes = [line.strip() for line in f.readlines()]
#
__font = cv2.FONT_HERSHEY_PLAIN
__colors = np.random.uniform(10, 255, size=(len(__classes), 3))  # possible outputs:
#
# __debug = True
# __log = False
#
# # loading the image
__img = cv2.imread("sample_image.jpeg")

#
# # -------------------  This is how I do it. ---------------------
# #       1. Load a bunch of images in in the main loop
# #       2. Visualize the bounding boxes
# #       3. Log out the bbox coordinate
#
#
layer_names = __network.getLayerNames()
print(f"layers_names: {layer_names}")
# layer_names = __network.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in __network.getUnconnectedOutLayers()]
print(f"output_layers: {output_layers}")
#
# img = cv2.resize(__img, None, fx=1, fy=1)
# height, width, channels = img.shape
#
# # Detecting objects
blob = cv2.dnn.blobFromImage(__img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
print(f'blob: {blob}')
for i, b in enumerate(blob):
    for c in b:
        for r in c:
            cv2.imshow(__win_name + str(i), c)


cv2.waitKey(0)
cv2.destroyAllWindows()
# This a simple test text to check the sound of keyboard typing.
# __network.setInput(blob)
# outputs = __network.forward(output_layers)
#
# # Showing information on the screen
# class_ids = []
# confidences = []
# bboxes = []
# for out in outputs:
#     for detection in out:
#         scores = detection[5:]
#         class_id = np.argmax(scores)
#         confidence = scores[class_id]
#         if confidence > 0.5:
#             # Object detected
#             center_x = int(detection[0] * width)
#             center_y = int(detection[1] * height)
#             w = int(detection[2] * width)
#             h = int(detection[3] * height)
#
#             # Rectangle coordinates
#             x = int(center_x - w / 2)
#             y = int(center_y - h / 2)
#
#             bboxes.append([x, y, w, h])
#             confidences.append(float(confidence))
#             class_ids.append(class_id)
#
# indices = cv2.dnn.NMSBoxes(bboxes, confidences, 0.4, 0.4)
#
# for i in range(len(bboxes)):
#     if i in indices:
#         x, y, w, h = bboxes[i]
#         if __log:
#             print(f"bounding box coordinates x: {x}, y: {y}, w: {w}, h: {h}")
#         label = str(__classes[class_ids[i]])
#         color = __colors[class_ids[i]]
#         cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
#         cv2.putText(img, label, (x, y + 30), __font, 3, color, 2)
#
# if __debug:
#     cv2.imshow("Image", img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
# if __log:
#     print(f"Other logs:\n")



