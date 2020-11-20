import cv2

from centerface import CenterFace


image = cv2.imread("images/demo.jpg")
h, w = image.shape[:2]

centerface = CenterFace(landmarks=True)
dets, lms = centerface(image, h, w, threshold=0.35)

for det in dets:
    boxes, score = det[:4], det[4]
    cv2.rectangle(
        image,
        (int(boxes[0]), int(boxes[1])),
        (int(boxes[2]), int(boxes[3])),
        (2, 255, 0),
        1,
    )

for lm in lms:
    for i in range(0, 5):
        cv2.circle(image, (int(lm[i * 2]), int(lm[i * 2 + 1])), 2, (0, 0, 255), -1)

cv2.imshow("", image)
cv2.waitKey(0)
