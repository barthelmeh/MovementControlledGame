import cv2
from ultralytics import YOLO


def main():
    capture = cv2.VideoCapture(-1)
    model = YOLO('yolov8m-pose.pt')

    if not capture.isOpened():
        print("Unable to open video capture")
        exit(-1)

    while capture.isOpened():
        ret, frame = capture.read()
        result = model.predict(frame, verbose=False)[0]

        img = result.plot()

        if not ret:
            break

        cv2.imshow('Camera', img)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
