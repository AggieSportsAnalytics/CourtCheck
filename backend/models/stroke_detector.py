import torch
import torch.nn as nn
import torchvision
import numpy as np
from torchvision import transforms
import imutils

# Stroke Recognition Classes
class Identity(nn.Module):
  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class FeatureExtractor(nn.Module):
  def __init__(self):
    super().__init__()
    self.feature_extractor = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1)
    self.feature_extractor.fc = Identity()

  def forward(self, x):
    output = self.feature_extractor(x)
    return output


class LSTM_model(nn.Module):
  """
  Time sequence model for stroke classifying
  """

  def __init__(
    self,
    num_classes,
    input_size=2048,
    num_layers=3,
    hidden_size=90,
    dtype=torch.cuda.FloatTensor,
  ):
    super().__init__()
    self.dtype = dtype
    self.input_size = input_size
    self.num_layers = num_layers
    self.hidden_size = hidden_size
    self.LSTM = nn.LSTM(
      input_size, hidden_size, num_layers, bias=True, batch_first=True
    )
    self.fc = nn.Linear(hidden_size, num_classes)

  def forward(self, x):
    # x shape is (batch_size, seq_len, input_size)
    h0, c0 = self.init_state(x.size(0))
    output, (hn, cn) = self.LSTM(x, (h0, c0))
    # size = 1
    size = x.size(1) // 4

    output = output[:, -size:, :]
    scores = self.fc(output.squeeze(0))
    # scores shape is (batch_size, num_classes)
    return scores

  def init_state(self, batch_size):
    return (
      torch.zeros(self.num_layers, batch_size, self.hidden_size).type(self.dtype),
      torch.zeros(self.num_layers, batch_size, self.hidden_size).type(self.dtype),
    )


class ActionRecognition:
  """
  Stroke recognition model
  """

  def __init__(self, model_saved_state, max_seq_len=55):
    self.dtype = (
      torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    )
    self.feature_extractor = FeatureExtractor()
    self.feature_extractor.eval()
    self.feature_extractor.type(self.dtype)
    self.normalize = transforms.Normalize(
      mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    self.max_seq_len = max_seq_len
    self.LSTM = LSTM_model(3, dtype=self.dtype)
    # Load model`s weights
    saved_state = torch.load(model_saved_state, map_location="cpu", weights_only=False)
    self.LSTM.load_state_dict(saved_state["model_state"])
    self.LSTM.eval()
    self.LSTM.type(self.dtype)
    self.frames_features_seq = None
    self.box_margin = 150
    self.softmax = nn.Softmax(dim=1)
    self.strokes_label = ["Forehand", "Backhand", "Service/Smash"]

  def predict_stroke(self, frame, player_box):
    """
    Predict the stroke for each frame
    """
    try:
      box_center = (
        int((player_box[0] + player_box[2]) / 2),
        int((player_box[1] + player_box[3]) / 2),
      )

      # Calculate patch boundaries
      y1 = max(0, int(box_center[1] - self.box_margin))
      y2 = min(frame.shape[0], int(box_center[1] + self.box_margin))
      x1 = max(0, int(box_center[0] - self.box_margin))
      x2 = min(frame.shape[1], int(box_center[0] + self.box_margin))

      # Check if patch is valid
      if y2 <= y1 or x2 <= x1:
        return None, "Unknown"

      patch = frame[y1:y2, x1:x2].copy()
      if patch.size == 0:
        return None, "Unknown"

      patch = imutils.resize(patch, 299)
      frame_t = patch.transpose((2, 0, 1)) / 255
      frame_tensor = torch.from_numpy(frame_t).type(self.dtype)
      frame_tensor = self.normalize(frame_tensor).unsqueeze(0)

      with torch.no_grad():
        features = self.feature_extractor(frame_tensor)
      features = features.unsqueeze(1)

      if self.frames_features_seq is None:
        self.frames_features_seq = features
      else:
        self.frames_features_seq = torch.cat(
          [self.frames_features_seq, features], dim=1
        )

      if self.frames_features_seq.size(1) > self.max_seq_len:
        remove = self.frames_features_seq[:, 0, :]
        remove.detach().cpu()
        self.frames_features_seq = self.frames_features_seq[:, 1:, :]

      with torch.no_grad():
        scores = self.LSTM(self.frames_features_seq)[-1].unsqueeze(0)
        probs = self.softmax(scores).squeeze().cpu().numpy()

      return probs, self.strokes_label[np.argmax(probs)]

    except Exception as e:
      print(f"Error in predict_stroke: {e}")
      return None, "Unknown"
    