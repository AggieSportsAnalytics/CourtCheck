# Tennis Bounds Project | CourtCheck
CourtCheck is a computer vision project aimed at detecting and tracking tennis court boundaries and ball movements. Utilizing advanced technologies such as YOLOv5, TrackNet, and Detectron2, this project integrates multiple models to deliver precise court detection and ball tracking capabilities.

## Workflow Overview
1. **Data Preparation:** 
    - Annotate images using OpenCVAT.
    - Organize datasets and annotations.
2. **Model Training:** 
    - Use Detectron2 for keypoint detection.
    - Train models incrementally on multiple games.
3. **Model Evaluation:** 
    - Evaluate the model using validation datasets.
    - Fine-tune the model based on performance.

## Installation
To set up the project, follow these steps:
```bash
git clone https://github.com/your-username/tennis-bounds-project.git
cd tennis-bounds-project
pip install -r requirements.txt
