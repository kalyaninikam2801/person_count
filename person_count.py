import cv2
import numpy as np
import time

# Load YOLOv3-tiny model
net = cv2.dnn.readNetFromDarknet("C:/AI projects/person_count_project/yolov3.cfg", "C:/AI projects/person_count_project/yolov3.weights")

# Get the output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class names (COCO dataset classes)
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize video capture (0 for webcam)
camera_stream_url = "http://192.168.0.101:8080/video"
cap = cv2.VideoCapture(camera_stream_url)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

previous_count = 0
smoothing_factor = 0.9  # Adjust this value for better smoothing

while True:
    start_time = time.time()  # Track FPS
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Lists to hold the detected bounding boxes, confidences, and class IDs
    class_ids = []
    confidences = []
    boxes = []

    # Loop over each detection
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3 and class_id == 0:  # Only 'person' class (0 is for person in COCO)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply NMS (Non-Maximum Suppression) to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.3)  # Adjusted thresholds

    # Track the count of detected persons
    person_count = 0
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw the bounding box
            label = str(classes[class_ids[i]]) + " " + str(round(confidences[i], 2))
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            person_count += 1

    # Debug: Print the person count only when detected persons change
    if person_count != previous_count:
        print(f"Person Count: {person_count}")
        previous_count = person_count

    # Smoothing the person count to avoid drastic fluctuations
    person_count = int(smoothing_factor * previous_count + (1 - smoothing_factor) * person_count)

    # Display the count
    cv2.putText(frame, f"Person Count: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # FPS Calculation and Display
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(frame, f"FPS: {round(fps, 2)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
