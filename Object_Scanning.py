import cv2
import numpy as np

# Load the pre-trained model
prototxt = 'MobileNetSSD_deploy.prototxt'  # Path to prototxt
model = 'MobileNetSSD_deploy.caffemodel'  # Path to caffemodel
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# Open a connection to the camera
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    print("Press 'q' to exit the video window.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If frame is read correctly, ret is True
        if not ret:
            print("Error: Could not read frame.")
            break

        # Prepare the frame for the model
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

        # Set the input to the pre-trained deep learning network
        net.setInput(blob)

        # Perform forward pass to get output of the output layers
        detections = net.forward()

        # Loop over the detections
        for i in range(detections.shape[2]):
            # Get the confidence of the prediction
            confidence = detections[0, 0, i, 2]

            # Filter out weak detections
            if confidence > 0.2:
                # Get the index of the class label
                idx = int(detections[0, 0, i, 1])
                
                # Define the list of class labels MobileNet SSD was trained to detect
                CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                           "dog", "horse", "motorbike", "person", "pottedplant",
                           "sheep", "sofa", "train", "tvmonitor"]

                # Get the class name
                class_name = CLASSES[idx]

                # Compute the (x, y)-coordinates of the bounding box
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")

                # Draw the bounding box
                label = "{}: {:.2f}%".format(class_name, confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Object Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()
