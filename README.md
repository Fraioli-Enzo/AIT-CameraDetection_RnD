# Fabric Defect Detection Using Computer Vision

This repository contains all the research and development related to a computer vision project focused on detecting defects in various types of fabrics.

To better understand the structure of this repository, here are a few explanations:

1. The `Images` folder contains the dataset used to train YOLO deep learning models.
2. The `py-src` folder includes many Python scripts that were used to test different approaches within this project.  
   The most important ones are:
   - `YOLO`: used to train YOLO models.
   - `YOLO_ImagesDetection`: used to analyze video or camera streams.
3. In the `YOLO_ImagesDetection` folder, the file `YOLO_ImagesDetection.py` is responsible for several tasks such as:
   - Detecting defects
   - Identifying the start and end points of fabric on a conveyor belt

For a more user-friendly experience, please refer to the `ComputerVision` repository.
