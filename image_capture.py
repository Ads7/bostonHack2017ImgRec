import cv2

cam = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_id = input('Enter Your ID: ')

samples = 0
while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        samples += 1
        cv2.imwrite("images/user_" + face_id + '_' + str(samples) + ".jpg",
                    gray[y: y + h, x: x + w])

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    if samples > 40:
        break


cam.release()
cv2.destroyAllWindows()

