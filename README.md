# CourtCheck Tennis Project 🎾

## 🏁 Automate tennis match analysis using the power of computer vision.

The CourtCheck Tennis Project leverages advanced computer vision techniques to accurately track tennis ball movements and court boundaries in tennis matches. This project aims to provide real-time insights and automated decision-making in tennis, reducing human error and enhancing the accuracy of in/out calls. The project integrates Python, machine learning, and computer vision to create a seamless and efficient system for tennis match analysis.

![https://github.com/AggieSportsAnalytics/CourtCheck/blob/main/images/game2_processed_10s.mp4)

## 🔑 Key Features

### Court Detection

The project employs keypoint detection algorithms to identify and track the tennis court's boundaries, ensuring accurate mapping and analysis of the court's dimensions.

- Uses Detectron2 for keypoint detection and court boundary identification.

<!-- Embedding video clip -->
<video width="320" height="240" controls>
  <source src="https://github.com/AggieSportsAnalytics/CourtCheck/blob/main/images/game2_processed_10s.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
**_Visual representation of the detected tennis court boundaries._**

### Ball Tracking

The system uses a deep learning model to track the tennis ball's position throughout the match, allowing for precise in/out call determinations.

- Utilizes a custom-trained TrackNet model for ball detection and tracking.

![ball-tracking-demo](https://github.com/SACUCD/CourtCheckTennis/assets/your-ball-tracking-image-url)
**_The yellow circle represents the detected position of the tennis ball._**

### 2D Court Simulation

One of the critical aspects of the project is transforming the detected ball and court positions onto a 2D map, enabling a clear and concise view of the ball's trajectory and court boundaries.

- Implements perspective transform using OpenCV.
- Transforms detected keypoints and ball positions to a 2D representation for easy analysis.

![2d-simulation-demo](https://github.com/SACUCD/CourtCheckTennis/assets/your-2d-simulation-image-url)
**_Red dots indicate transformed keypoints on the 2D court simulation._**

## 🪴 Areas of Improvement

- **Accuracy**: Enhance the accuracy of ball and court detection to ensure reliable analysis.
- **Real-Time Processing**: Implement real-time video feed analysis for instant decision-making during matches.
- **Automatic Court Detection**: Automate the court detection process to handle different court types and angles without manual input.
- **Player Tracking**: Integrate player tracking to provide comprehensive match statistics and insights.

## 🚀 Further Uses

- **Match Analytics**: Utilize the system for detailed match analytics, including player performance and shot accuracy.
- **Training and Coaching**: Provide coaches and players with valuable data to improve training sessions and match strategies.
- **Broadcast Enhancement**: Enhance sports broadcasting with real-time analysis and insights for viewers.

## 💻 Technology

- **OpenCV**: For image and video processing.
- **Detectron2**: For keypoint detection and court boundary identification.
- **TrackNet**: For tennis ball detection and tracking.
- **NumPy**: For numerical computations and data manipulation.
- **PyTorch**: For building and training machine learning models.
- **tqdm**: For progress bar visualization in loops and processing tasks.

## Installation

To set up the project, clone the repository and install the dependencies using the `requirements.txt` file:

```bash
git clone https://github.com/SACUCD/CourtCheckTennis.git
cd CourtCheckTennis
pip install -r requirements.txt
```
