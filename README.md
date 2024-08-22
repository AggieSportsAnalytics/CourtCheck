
# CourtCheck <img src="https://github.com/AggieSportsAnalytics/CourtCheck/blob/cory/images/courtcheck_ball_logo.png" alt="CourtCheck Logo" style="width: 80px; vertical-align: middle;"> 

### 🏁 Automate tennis match analysis using the power of computer vision.

CourtCheck leverages advanced computer vision techniques to accurately track tennis ball movements and court boundaries in tennis matches. This project aims to provide real-time insights and automated decision-making in tennis, reducing human error and enhancing the accuracy of in/out calls. CourtCheck integrates Python, machine learning, and computer vision to create a seamless and efficient system for tennis match analysis.

![courtcheck-demo](https://github.com/AggieSportsAnalytics/CourtCheck/blob/main/images/game2_processed_10s.gif)

# 🔑 Process & Key Features



## 1. 🔎 Court Detection

CourtCheck employs keypoint detection algorithms to identify and track the tennis court's boundaries, ensuring accurate mapping and analysis of the court's dimensions.

### a. 📑 Annotation

We began by annotating images using OpenCV in the COCO format, generating JSON files for each annotated image. The [OpenCV Annotation Tool](https://app.cvat.ai/) provides an excellent interface for image annotation and export in various formats. It also features an interpolation tool that allows the use of a skeleton to label key frames, which can be interpolated over consecutive frames in the video.

![annotation-demo](https://github.com/AggieSportsAnalytics/CourtCheck/blob/cory/images/opencv_annotation.gif)

You can find the annotations [here](https://drive.google.com/drive/folders/16HugQeDoXUX420yKjg7pGVu3NG6linMV?usp=drive_link).

Each label in the skeleton represents a keypoint on the tennis court, identifying an important corner or intersection of lines that are crucial for the overall court detection when training the model. Here are the keypoints and their corresponding labels:

| Keypoint | Label                       | Keypoint | Label                           | Keypoint | Label                       |
|----------|-----------------------------|----------|---------------------------------|----------|-----------------------------|
| BTL      | Bottom Top Left             | ITM      | Inner Top Middle                | ITR      | Inner Top Right             |
| BTLI     | Bottom Top Left Inner       | IBR      | Inner Bottom Right              | NL       | Net Left                    |
| BTRI     | Bottom Top Right Inner      | NR       | Net Right                       | BBL      | Bottom Bottom Left          |
| BTR      | Bottom Top Right            | NM       | Net Middle                      | IBL      | Inner Bottom Left           |
| BBR      | Bottom Bottom Right         | ITL      | Inner Top Left                  | IBM      | Inner Bottom Middle         |
| BBRI     | Bottom Bottom Right Inner   |          |                                 | BBLI     | Bottom Bottom Left Inner    |

### b. 🤖 Training the Model

We utilized the A100 Nvidia GPU on Google Colab to train our Detectron2 model on different types of datasets. These datasets included varying court surfaces and slightly different camera angles to ensure robustness and generalizability of the model. Below, we explain the process and provide the code used for training the model incrementally with mixed datasets.

Below is an overview of the Detectron2 architecture:

<div align="center">
    <img src="https://github.com/AggieSportsAnalytics/CourtCheck/blob/cory/images/detectron2_architecture.png" alt="Detectron2 Architecture" width="700"/>
</div>
<br>

We used the `COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml` configuration file because it is specifically designed for keypoint detection tasks. The [keypoint_rcnn_R_50_FPN_3x.yaml](https://drive.google.com/drive/folders/18t8oUo5_jzYYD1vnFzjLw7uhxmagXINX?usp=drive_link) configuration is well-suited for this task because it includes a pre-trained ResNet-50 backbone that provides strong feature extraction capabilities, coupled with a Feature Pyramid Network (FPN) that helps detect objects at multiple scales. This combination ensures that the model can accurately identify and track the key points on the tennis court, providing precise court boundary detection and enabling accurate in/out call determinations.


#### c. 🧬 Model Code

The code below sets up and trains the Detectron2 model using multiple datasets:

1. **Dataset Registration**: Registers the training and validation datasets.
2. **COCO Instance Registration**: Registers the datasets in COCO format.
3. **Metadata Configuration**: Configures metadata for keypoints, keypoint flip map, and skeleton.
4. **Configuration Setup**: Sets up the model configuration, including dataset paths, data loader workers, batch size, learning rate, maximum iterations, learning rate decay steps, and checkpoint period.
5. **Trainer Initialization and Training**: Initializes a custom trainer and starts or resumes the training process.

You can find the Google Colab Notebook [here](https://colab.research.google.com/drive/1huJ4f0yOApwM4NR8gpXIHktHkTgrbL_m?usp=drive_link).

```python
# Function to set up and train the model with mixed datasets incrementally
def train_model(max_iter, resume=False):
    register_datasets(train_json_files, train_image_dirs, "tennis_game_train")
    register_datasets(val_json_files, val_image_dirs, "tennis_game_val")

    register_coco_instances(f"tennis_game_train", {}, train_json_files, train_image_dirs)
    register_coco_instances(f"tennis_game_val", {}, val_json_files, val_image_dirs)

    MetadataCatalog.get(f"tennis_game_train").keypoint_names = keypoint_names
    MetadataCatalog.get(f"tennis_game_train").keypoint_flip_map = keypoint_flip_map
    MetadataCatalog.get(f"tennis_game_train").keypoint_connection_rules = skeleton

    MetadataCatalog.get(f"tennis_game_val").keypoint_names = keypoint_names
    MetadataCatalog.get(f"tennis_game_val").keypoint_flip_map = keypoint_flip_map
    MetadataCatalog.get(f"tennis_game_val").keypoint_connection_rules = skeleton

    cfg = get_cfg()
    cfg.merge_from_file("/content/drive/MyDrive/ASA Tennis Bounds Project/models/court_detection_model/detectron2/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
    cfg.DATASETS.TRAIN = tuple([os.path.basename(f).split('.')[0] for f in train_json_files])
    cfg.DATASETS.TEST = tuple([os.path.basename(f).split('.')[0] for f in val_json_files])
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.SOLVER.IMS_PER_BATCH = 4  # Increase if you have more GPU memory
    cfg.SOLVER.BASE_LR = 0.0001  # Lower learning rate for more careful training
    cfg.SOLVER.MAX_ITER = max_iter  # Total number of iterations
    cfg.SOLVER.STEPS = [int(max_iter*0.75), int(max_iter*0.875)]  # Decay learning rate
    cfg.SOLVER.GAMMA = 0.1  # Decay factor
    cfg.SOLVER.CHECKPOINT_PERIOD = 20000  # Save a checkpoint every 20000 iterations
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  # Increase for more stable gradients
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 11  # Your dataset has 11 classes

    output_dir = f"/content/drive/MyDrive/ASA Tennis Bounds Project/models/court_detection_model/detectron2/game_model"
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = TrainerWithEval(cfg)
    trainer.resume_or_load(resume=resume)
    trainer.train()

# Training parameters
custom_iter = 50000  # Adjust this to your custom number of iterations per session
max_iter = last_iter + custom_iter  # Change this for the number of iterations per session

# Execute to train model
train_model(max_iter, resume=True)
```

## 2. 📽️ Post Processing

After training the model, the next crucial step is post-processing the results to ensure accurate and meaningful outputs. Post-processing involves refining the model's predictions and visualizing the detected key points on the tennis court for better interpretation and analysis.

### a. 🔲 Visualizing the Court on the Main Frame

To accurately visualize the tennis court on the main video frame, we start by detecting key points on the court using the trained model. These key points correspond to specific locations on the court, such as the corners and intersections of lines. Visualizing these key points on the frame helps us understand how well the model is detecting the court's structure.

#### i. Extracting Key Points from the Model

The court detection model `(court_predictor)` outputs instances that include predicted key points. These key points are stored in an array where each element corresponds to a specific point on the court, identified by its (x, y) coordinates.

Here's how the key points are extracted:
```
outputs = court_predictor(img)
instances = outputs["instances"]

if len(instances) > 0:
    keypoints = instances.pred_keypoints.cpu().numpy()[0]
else:
    keypoints = np.zeros((17, 3))
```
- `outputs["instances"]`: This contains all detected instances in the frame, including the detected court.
- `instances.pred_keypoints.cpu().numpy()[0]`: This extracts the key points of the detected court. The key points are converted from a tensor to a numpy array for further processing.

If no key points are detected, a default array of zeros is used to avoid errors in subsequent processing.

#### i. Visualizing Key Points and Court Lines

Once the key points are extracted, the next step is to visualize them by drawing polylines between the points that align with the court lines and boundaries. These lines help in creating a clear and precise representation of the tennis court structure. Here are the specific polylines to be drawn:

```python
lines = [
    ("BTL", "BTLI"), ("BTLI", "BTRI"), ("BTL", "NL"), ("BTLI", "ITL"),
    ("BTRI", "BTR"), ("BTR", "NR"), ("BTRI", "ITR"), ("ITL", "ITM"), ("ITM", "ITR"),
    ("ITL", "IBL"), ("ITM", "NM"), ("ITR", "IBR"), ("NL", "NM"), ("NL", "BBL"),
    ("NM", "IBM"), ("NR", "BBR"), ("NM", "NR"), ("IBL", "IBM"),
    ("IBM", "IBR"), ("IBL", "BBLI"), ("IBR", "BBRI"), ("BBR", "BBRI"),
    ("BBRI", "BBLI"), ("BBL", "BBLI"),
]
```
Each pair in the `lines` list represents two key points between which a line will be drawn. These lines outline the court's structure on the video frame.

The `visualize_predictions` function is essential for visualizing model predictions on an input image. Here are two key parts of the function:

```python
outputs = predictor(img)
v = Visualizer(
    img[:, :, ::-1],
    metadata=MetadataCatalog.get("tennis_game_train"),
    scale=0.8,
    instance_mode=ColorMode.IMAGE,
)
instances = outputs["instances"].to("cpu")

if len(instances) > 0:
    max_conf_idx = instances.scores.argmax()
    instances = instances[max_conf_idx : max_conf_idx + 1]

out = v.draw_instance_predictions(instances)
keypoints = instances.pred_keypoints.numpy()[0]
```
- `Visualizer`: This tool is used to overlay the detected key points and lines on the original image.
- `draw_instance_predictions`: This function draws the visual elements on the frame, including key points and the connecting lines.

This process results in a visual overlay of the court on the original video frame, allowing for immediate visual verification of the court detection accuracy.

To ensure that the detected key points on the tennis court are stable and less jittery, especially when dealing with video frames, we use a stabilization technique. This involves averaging the positions of detected key points over a history of frames.

Key Point History Initialization
```python
keypoint_history = {name: deque(maxlen=10) for name in keypoint_names}
```
We initialize a dictionary called keypoint_history where each key is a key point name, and the value is a deque (double-ended queue) with a maximum length of 10. This deque will store the positions of each key point over the last 10 frames.

Stabalizing Keypoints
```python
def stabilize_points(keypoints):
    stabilized_points = []
    for i, keypoint in enumerate(keypoints):
        keypoint_history[keypoint_names[i]].append(keypoint[:2])
        if len(keypoint_history[keypoint_names[i]]) > 1:
            stabilized_points.append(
                np.mean(np.array(keypoint_history[keypoint_names[i]]), axis=0)
            )
        else:
            stabilized_points.append(keypoint[:2])
    return np.array(stabilized_points)
```
The `stabilize_points` function then uses the `keypoint_history` dictionary to process the detected key points and reduce jitter by averaging their positions over the last 10 frames. For each detected key point, its position is appended to the corresponding deque in the `keypoint_history` dictionary. If the deque contains more than one position, the average of these positions is computed and added to the `stabilized_points` list. If the deque contains only one position, the key point is added to the list as is. This results in more consistent and smooth key point positions for further processing and visualization.

### b. 📐 Transforming the Court into a 2D Plane Using Homography

After visualizing the court on the main frame, the next step is to transform these detected key points into a 2D, top-down view of the court. This transformation is essential for accurate analysis of the ball's position in relation to the court lines.

#### i. Homography Transformation

Homography Transformation is a mathematical technique used to map points from one plane to another, such as transforming the court from the camera’s perspective view to a top-down 2D view. In this project, homography transformation is crucial because it allows us to create an accurate 2D representation of the court, which is necessary for determining ball positions and making in/out calls.

#### ii. Extracting and Preparing Data for Transformation

The process starts with extracting the key points as discussed earlier. These key points are then used to define the court’s boundaries in both the original perspective and the target 2D plane.

```python
src_points = np.array(
    [
        keypoint_dict["BTL"],  # Bottom-Top-Left
        keypoint_dict["BTR"],  # Bottom-Top-Right
        keypoint_dict["BBL"],  # Bottom-Bottom-Left
        keypoint_dict["BBR"],  # Bottom-Bottom-Right
    ],
    dtype=np.float32,
)

dst_points = np.array(
    [
        [black_frame_width // 6, black_frame_height // 7],  # BTL in 2D
        [black_frame_width * 5 // 6, black_frame_height // 7],  # BTR in 2D
        [black_frame_width // 6, black_frame_height * 6 // 7],  # BBL in 2D
        [black_frame_width * 5 // 6, black_frame_height * 6 // 7],  # BBR in 2D
    ],
    dtype=np.float32,
)
```
- `src_points`: These are the coordinates of the key points in the original frame, representing the four corners of the court.
- `dst_points`: These are the coordinates where these points should be mapped in the 2D plane. They represent where the corners of the court should be in the top-down view.

#### iii. Computing the Homography Matrix

The homography matrix is then computed using the source and destination points:
```python
matrix = cv2.getPerspectiveTransform(src_points, dst_points)
transformed_keypoints = cv2.perspectiveTransform(keypoints[None, :, :2], matrix)[0]
```
- `cv2.getPerspectiveTransform`: This function calculates the homography matrix that maps src_points to dst_points.
- `cv2.perspectiveTransform`: This function applies the homography matrix to the key points, transforming them from the original perspective view to the 2D plane.

The result is a set of transformed key points that represent the court in a top-down 2D view.

#### iv. Visualizing the 2D Court

Finally, the transformed key points are used to visualize the court in 2D:

```python
court_skeleton = visualize_2d(
    transformed_keypoints, lines, black_frame_width, black_frame_height
)
```

This function draws the court lines and boundaries based on the transformed key points, providing a clear and accurate 2D representation of the court.
| Court Detection in Main Frame | Transposed 2D Plane |
|:-----------------------------:|:-------------------:|
| ![Court Detection in Main Frame](https://github.com/AggieSportsAnalytics/CourtCheck/blob/main/images/game1_court_processed.gif) | <img src="https://github.com/AggieSportsAnalytics/CourtCheck/blob/main/images/game2_2Dskeleton_10s.gif" alt="Transposed 2D Plane" style="width: 50%;"> |

This 2D transformation is a crucial step in ensuring the accuracy and effectiveness of the CourtCheck system, enabling precise in/out calls and detailed match analysis.

## 🎾 Ball Tracking

CourtCheck is integrated with a ball tracking model called TrackNet. TrackNet is an advanced deep learning model specifically designed for tracking tennis balls, and it has been instrumental in enhancing the overall functionality of this project. The TrackNet model used here is adapted from [yastrebksv's TrackNet implementation on GitHub](https://github.com/yastrebksv/TrackNet).

### ↔️ Integration Process

The integration of court detection with TrackNet involves the following steps:

1. **Court Detection**: First, the court detection model processes each frame of the video to identify and map the court's key points. This ensures that the court is accurately detected and transformed into a 2D plane.

2. **Ball Tracking with TrackNet**: The TrackNet model is then applied to the same frames to detect and track the tennis ball. TrackNet's robust tracking capability allows for precise ball movement analysis throughout the video.

3. **Combining Results**: The results from both models are combined to provide a comprehensive visualization of the tennis game. This includes the detected court and the tracked ball, offering insights into ball positions relative to the court boundaries.

The `combine_results` function in `process_video.py` integrates the outputs from `court_detection.py` and `ball_detection.py` to process each video frame and produce the final visualization.

Here are a couple key steps of the function

1. **Court Detection**:
    ```python
    outputs = court_predictor(frame)
    instances = outputs["instances"]
    if len(instances) > 0:
        keypoints = instances.pred_keypoints.cpu().numpy()[0]
        keypoints_found += 1
        processed_frame = visualize_predictions(frame, court_predictor, keypoint_names, lines, black_frame_width, black_frame_height)
    else:
        keypoints = np.zeros((17, 3))
        processed_frame = frame.copy()
    ```
    The court detection model is applied to the current frame. If keypoints are found, they are visualized on the frame; otherwise, the frame is copied as is.

2. **Ball Tracking**:
    ```python
    x_pred, y_pred = detect_ball(tracknet_model, device, frame, prev_frame, prev_prev_frame)
    ball_track.append((x_pred, y_pred))
    ```
    The TrackNet model is used to detect the ball's position in the current frame, and the position is added to the `ball_track` list.

3. **Visualizing Ball Movement**:
    ```python
    if x_pred and y_pred:
        for j in range(min(7, len(ball_track))):
            if ball_track[-j][0] is not None and ball_track[-j][1] is not None:
                if 0 <= int(ball_track[-j][0]) < processed_frame.shape[1] and 0 <= int(ball_track[-j][1]) < processed_frame.shape[0]:
                    cv2.circle(processed_frame, (int(ball_track[-j][0]), int(ball_track[-j][1])), max(2, 7 - j), (255, 255, 0), -1)
    ```
    The detected ball positions are visualized on the frame, drawing circles to represent the ball's trajectory.

4. **Stabilizing and Transforming Key Points**:
    ```python
    stabilized_points = stabilize_points(keypoints)
    transformed_keypoints, matrix = transform_points(stabilized_points, black_frame_width, black_frame_height)
    court_skeleton = visualize_2d(transformed_keypoints, lines, black_frame_width, black_frame_height)
    ```
    Key points are stabilized and transformed to fit within a defined black frame, and the court skeleton is visualized.

5. **Transforming Ball Position to 2D Plane**:
    ```python
    if x_pred and y_pred:
        ball_pos_2d = transform_ball_2d(x_pred, y_pred, matrix)
        if 0 <= int(ball_pos_2d[0]) < court_skeleton.shape[1] and 0 <= int(ball_pos_2d[1]) < court_skeleton.shape[0]:
            cv2.circle(court_skeleton, (int(ball_pos_2d[0]), int(ball_pos_2d[1])), 3, (255, 255, 0), -1)
    ```
    The ball's position is transformed into the 2D plane of the court skeleton and visualized.
   
    ![ball-demo](https://github.com/AggieSportsAnalytics/CourtCheck/blob/cory/images/game2_2Dskeleton_court_ball.gif)
   

7. **Combining and Storing the Processed Frame**:
    ```python
    processed_frame[0 : court_skeleton.shape[0], 0 : court_skeleton.shape[1]] = court_skeleton
    combined_frames.append(processed_frame)
    ```
    The court skeleton is overlaid onto the processed frame, and the frame is added to the list of combined frames.

The `combine_results` function effectively integrates court detection and ball tracking to produce a comprehensive visualization of the tennis game, showing both the court and the ball's movement in each frame.

![demo](https://github.com/AggieSportsAnalytics/CourtCheck/blob/main/images/game9_processed_10s.gif)

#### Crediting TrackNet

The TrackNet model used in this project is credited to [yastrebksv](https://github.com/yastrebksv/TrackNet). Their implementation provided the foundation for the ball tracking functionality integrated into this project.


# 🪴 Areas of Improvement

- **Accuracy**: Enhance the accuracy of ball and court detection to ensure reliable analysis.
- **Real-Time Processing**: Implement real-time video feed analysis for instant decision-making during matches.
- **Automatic Court Detection**: Automate the court detection process to handle different court types and angles without manual input.
- **Player Tracking**: Integrate player tracking to provide comprehensive match statistics and insights.

# 🚀 Further Uses

- **Match Analytics**: Utilize the system for detailed match analytics, including player performance and shot accuracy.
- **Training and Coaching**: Provide coaches and players with valuable data to improve training sessions and match strategies.
- **Broadcast Enhancement**: Enhance sports broadcasting with real-time analysis and insights for viewers.

# 💻 Technology

- **OpenCV**: For image and video processing.
- **Detectron2**: For keypoint detection and court boundary identification.
- **TrackNet**: For tennis ball detection and tracking.
- **NumPy**: For numerical computations and data manipulation.
- **PyTorch**: For building and training machine learning models.
- **tqdm**: For progress bar visualization in loops and processing tasks.

## 🛠️ Installation

To set up the project, clone the repository and install the dependencies using the `requirements.txt` file:

```bash
git clone https://github.com/AggieSportsAnalytics/CourtCheck.git
cd CourtCheck
pip install -r requirements.txt
```
After installing the requirements, navigate to the `scripts/process_video.py` directory. At the bottom of the script, update the video paths to your desired input and output locations:

```
video_path = "..."  # Input Video Path (mp4 format)
output_path = "..."  # Output Video Path (mp4 format)
```

⚠️ Note: This process is for post-processing, meaning it will infer on a video input to detect the court and the ball. This operation is intensive on computer hardware and may take quite a bit of time to complete. If you want to use Google Colab's online GPU/processor instead, then head to [Google Colab Notebook](https://colab.research.google.com/drive/11wkF5_nDDkTFaKCEX-e7doO-I55bn3NW?usp=sharing). Make sure to run all cells in chronological order.


