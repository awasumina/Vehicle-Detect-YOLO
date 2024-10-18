# YOLO Object Detection with Python

This project demonstrates how to use the YOLOv8 model for object detection. YOLO is used to detect objects in an image and draw bounding boxes with labels around them. The detected objects, their coordinates, and confidence scores are displayed.


## Model

We use the pre-trained YOLOv8 model (`yolov8m.pt`) from the [Ultralytics](https://github.com/ultralytics/ultralytics) library for object detection.

## Code Explanation

1. **Loading the YOLOv8 model:**

   We first import the necessary libraries and load the YOLOv8 medium model (`yolov8m.pt`) using the `Ultralytics` library.

   ```python
   from ultralytics import YOLO
   model = YOLO('yolov8m.pt')
   ```

2. **Predicting Objects:**

   We pass an image to the model for detection. The image is loaded using the Python Imaging Library (PIL) and processed by the model to detect objects, returning their bounding box coordinates, class labels, and confidence scores.

   ```python
   url = "/kaggle/input/images/R.jpg"
   img = Image.open(url)
   results = model.predict(url)
   result = results[0]
   ```

3. **Displaying Detected Objects:**

   The results from the model include the detected objects, their coordinates, and confidence levels. We display this information for each detected object.

   ```python
   for box in result.boxes:
       cords = box.xyxy[0].tolist()
       class_id = box.cls[0].item()
       conf = box.conf[0].item()
       print(f"Object type: {result.names[class_id]}")
       print(f"Coordinates: {cords}")
       print(f"Probability: {conf}")
   ```

4. **Drawing Bounding Boxes:**

   We use OpenCV to draw bounding boxes around the detected objects on the image. The bounding box is drawn using the coordinates, and the class label is displayed near the box.

   ```python
   for box in result.boxes:
       cords = box.xyxy[0].tolist()
       start = (int(cords[0]), int(cords[1]))
       end = (int(cords[2]), int(cords[3]))
       cv2.rectangle(image, start, end, (0, 200, 0), thickness=2)
       cv2.putText(image, result.names[class_id], (start[0]+15, start[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 0, 10), 2)
   ```

5. **Showing the Processed Image:**

   Finally, the image with bounding boxes and labels is displayed using `Image.fromarray(image)`.

   ```python
   Image.fromarray(image)
   ```

## Example Output

For an input image, the output might look like:

```
Object type: car (2.0)
Coordinates: [627.65, 1070.49, 1298.04, 1270.12]
Probability: 0.92
------------------
Object type: truck (7.0)
Coordinates: [28.05, 990.92, 543.95, 1270.1]
Probability: 0.87
------------------
```

The processed image will have bounding boxes and labels drawn around the detected objects.
   