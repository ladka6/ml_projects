# Lane Detection (Classical Computer Vision)

## 🎯 Objective
Implement lane detection in dashcam footage using classical computer vision techniques (OpenCV).  
This is a foundational task in autonomous driving, helping the vehicle stay within lane boundaries.

## 📂 Dataset
- **Name:** Udacity Self-Driving Car Lane Lines Dataset
- **Link:** https://github.com/udacity/CarND-LaneLines-P1
- **Content:** Road images and videos with visible lane markings.
- **Download Instructions:**  
```bash
git clone https://github.com/udacity/CarND-LaneLines-P1.git data
```

## 🛠️ Methods
- Color space filtering (HLS, HSV)
- Canny edge detection
- Hough line transform
- Perspective transform for lane curvature

## 🚀 How to Run
```bash
python lane_detection.py --input data/solidWhiteRight.mp4 --output output.mp4
```

## 📊 Results
*(Add before/after images and sample video link)*

## 📚 References
- OpenCV Documentation: https://docs.opencv.org/
- Udacity Self-Driving Car Nanodegree Lane Lines Project
