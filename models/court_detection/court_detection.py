from dependencies import *

# Set the session number


# Function to unregister the dataset if it already exists
def unregister_dataset(dataset_name):
    if dataset_name in DatasetCatalog.list():
        DatasetCatalog.pop(dataset_name)
        MetadataCatalog.pop(dataset_name)


# Register the datasets
def register_datasets(json_files, image_dirs, dataset_name):
    for json_file, image_dir in zip(json_files, image_dirs):
        dataset_name = os.path.basename(json_file).split(".")[0]
        unregister_dataset(dataset_name)
        register_coco_instances(dataset_name, {}, json_file, image_dir)
        MetadataCatalog.get(dataset_name).keypoint_names = keypoint_names
        MetadataCatalog.get(dataset_name).keypoint_flip_map = keypoint_flip_map
        MetadataCatalog.get(dataset_name).keypoint_connection_rules = skeleton
        print(f"Registered dataset {dataset_name} with {json_file} and {image_dir}")


register_datasets(train_json_files, train_image_dirs, "tennis_game_train")
register_datasets(val_json_files, val_image_dirs, "tennis_game_val")


# Custom trainer with evaluator
class TrainerWithEval(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)


# Function to set up and train the model with mixed datasets incrementally
def train_model(max_iter, resume=False):
    register_datasets(train_json_files, train_image_dirs, "tennis_game_train")
    register_datasets(val_json_files, val_image_dirs, "tennis_game_val")

    # register_coco_instances(f"tennis_game_train", {}, train_json_files, train_image_dirs)
    # register_coco_instances(f"tennis_game_val", {}, val_json_files, val_image_dirs)

    # MetadataCatalog.get(f"tennis_game_train").keypoint_names = keypoint_names
    # MetadataCatalog.get(f"tennis_game_train").keypoint_flip_map = keypoint_flip_map
    # MetadataCatalog.get(f"tennis_game_train").keypoint_connection_rules = skeleton

    # MetadataCatalog.get(f"tennis_game_val").keypoint_names = keypoint_names
    # MetadataCatalog.get(f"tennis_game_val").keypoint_flip_map = keypoint_flip_map
    # MetadataCatalog.get(f"tennis_game_val").keypoint_connection_rules = skeleton

    cfg = get_cfg()
    cfg.merge_from_file(
        "/content/drive/MyDrive/ASA Tennis Bounds Project/models/court_detection_model/detectron2/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.DATASETS.TRAIN = tuple(
        [os.path.basename(f).split(".")[0] for f in train_json_files]
    )
    cfg.DATASETS.TEST = tuple(
        [os.path.basename(f).split(".")[0] for f in val_json_files]
    )
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.SOLVER.IMS_PER_BATCH = 4  # Increase if you have more GPU memory
    cfg.SOLVER.BASE_LR = 0.0001  # Lower learning rate for more careful training
    cfg.SOLVER.MAX_ITER = max_iter  # Total number of iterations
    cfg.SOLVER.STEPS = [
        int(max_iter * 0.75),
        int(max_iter * 0.875),
    ]  # Decay learning rate
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


# Function to print the session number
def print_session_number(session_number):
    print(f"Current Session Number: {session_number}")


# Function to print the last iteration where the model left off
def print_last_iteration(output_dir, last_checkpoint):
    checkpoint_path = os.path.join(output_dir, last_checkpoint)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        last_iter = checkpoint.get("iteration", "unknown")
        print(f"Last Iteration: {last_iter + 1}")
    else:
        print("No previous checkpoint found.")
        last_iter = 0
    return last_iter


# Set the session number and iteration intervals
output_dir = "/content/drive/MyDrive/ASA Tennis Bounds Project/models/court_detection_model/detectron2/game_model"
last_checkpoint = "model_final.pth"

# Print the session number and last iteration
print_session_number(session_number)
last_iter = print_last_iteration(output_dir, last_checkpoint)

# Training parameters
custom_iter = 50000  # Adjust this to your custom number of iterations per session
max_iter = (
    last_iter + custom_iter
)  # Change this for the number of iterations per session


# Execute to train model
train_model(max_iter, resume=True)
