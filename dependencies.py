# dependencies.py

# Detectron2 imports
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import Instances
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator

# Other imports
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import numpy as np
import logging
import os
import json
import random
import cv2
import torch
import torchvision
import time

# Google Colab imports
try:
    from google.colab import drive

    drive.mount("/content/drive")

    from google.colab.patches import cv2_imshow
except ImportError:
    pass

# Any other shared setup or configuration can go here
