The Person Counting Project is designed to count the number of people in a given area using computer vision. It uses a live camera feed (webcam or IP camera) to detect and track individuals in real time. This project is useful for applications like monitoring room occupancy, ensuring social distancing, managing event security, or optimizing resource allocation in crowded spaces.

Project Objectives
Real-Time Person Detection: Identify individuals in the camera feed using pre-trained object detection models.
Accurate Person Counting: Maintain a count of people entering and exiting a defined area.
Customizability: Allow users to adjust settings for different environments and camera setups.
Workflow
Input Video Stream:

Capture live video from a webcam or an IP camera.
Ensure proper resolution and frame rate for real-time processing.
Object Detection:

Use a pre-trained deep learning model (e.g., YOLOv4, SSD, or MobileNet-SSD) to detect persons in each frame.
Filter out non-human objects using the model's class labels.
Tracking:

Assign a unique ID to each detected individual.
Use tracking algorithms like SORT (Simple Online and Realtime Tracking) or OpenCV's built-in tracking API to track people across frames.
Counting Logic:

Define a region of interest (ROI), such as an entrance or exit.
Count individuals when they cross the ROI in a specific direction.
Update the total count dynamically.
Output:

Display the live video feed with bounding boxes around detected persons.
Overlay the real-time person count on the video feed.
