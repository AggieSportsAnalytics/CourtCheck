import ultralytics

class HumanTracker:
    """
    Uses a YOLO model from the ultralytics library to detect humans (or specified classes) in an image.
    """

    def __init__(self, model_path, class_no=[0]):
        """
        Initialize the HumanTracker.
        
        :param model_path: Path to the YOLO model weights.
        :param class_no: List of class numbers to track (default is [0] for humans).
        """
        self.__model = ultralytics.YOLO(model_path)
        self.__class_number = class_no
        self.__results = None

    def track(self, image):
        """
        Process the input image and return bounding boxes for detected objects matching the specified classes.
        
        :param image: The image in which to detect objects.
        :return: A list of tuples, each containing two coordinate pairs (top-left and bottom-right).
        """
        self.__results = self.__model.predict(image, classes=self.__class_number)
        self.__returns = []  # Initialize returns as a list

        for result in self.__results:
            for box in result.boxes:
                bbox = box.xyxy[0] if hasattr(box, 'xyxy') else None

                if bbox is not None:
                    x1, y1, x2, y2 = map(int, bbox)
                    self.__returns.append(((x1, y1), (x2, y2)))

        return self.__returns
