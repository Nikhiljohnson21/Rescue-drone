import cv2
import numpy as np

# Load pre-trained MobileNet SSD model for human detection
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

# Open a video capture object
cap = cv2.VideoCapture(0)  # Replace with your video file

# Initialize the total count and total confidence variables
total_people = 0
total_confidence = 0.0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("End of video, exiting...")
        break

    # Resize frame for better processing speed (optional)
    frame_resized = cv2.resize(frame, (300, 300))

    # Convert the frame to a blob
    blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), 127.5)

    # Set the input to the neural network
    net.setInput(blob)

    # Perform inference and get the detection results
    detections = net.forward()

    # Reset the total count and total confidence for each frame
    total_people = 0
    total_confidence = 0.0

    # Loop over the detections and draw bounding boxes around humans
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2 and int(detections[0, 0, i, 1]) == 15:  # Class 15 corresponds to 'person'
            # Increment the total count for each detected person
            total_people += 1

            # Accumulate confidence scores for all detected people
            total_confidence += confidence

            # Calculate coordinates for drawing the bounding box
            (h, w) = frame.shape[:2]
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Draw bounding box around the detected person
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Calculate average confidence (if at least one person is detected)
    average_confidence = total_confidence / max(1, total_people)

    # Display the total count and average confidence of people in the frame
    cv2.putText(frame, f'Total People: {total_people}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Average Confidence: {average_confidence:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
