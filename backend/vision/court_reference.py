## Court Reference
import cv2
import numpy as np

class CourtReference:
  """
  Canonical tennis court reference in pixel space.
  Used for homography mapping and minimap visualization.
  """

  def __init__(self):
    self.baseline_top = ((286, 561), (1379, 561))
    self.baseline_bottom = ((286, 2935), (1379, 2935))
    self.net = ((286, 1748), (1379, 1748))
    self.left_court_line = ((286, 561), (286, 2935))
    self.right_court_line = ((1379, 561), (1379, 2935))
    self.left_inner_line = ((423, 561), (423, 2935))
    self.right_inner_line = ((1242, 561), (1242, 2935))
    self.middle_line = ((832, 1110), (832, 2386))
    self.top_inner_line = ((423, 1110), (1242, 1110))
    self.bottom_inner_line = ((423, 2386), (1242, 2386))
    self.top_extra_part = (832.5, 580)
    self.bottom_extra_part = (832.5, 2910)

    self.key_points = [
        *self.baseline_top,
        *self.baseline_bottom,
        *self.left_inner_line,
        *self.right_inner_line,
        *self.top_inner_line,
        *self.bottom_inner_line,
        *self.middle_line,
    ]

    self.border_points = [*self.baseline_top, *self.baseline_bottom[::-1]]

    self.court_conf = {
        1: [*self.baseline_top, *self.baseline_bottom],
        2: [
            self.left_inner_line[0],
            self.right_inner_line[0],
            self.left_inner_line[1],
            self.right_inner_line[1],
        ],
        3: [
            self.left_inner_line[0],
            self.right_court_line[0],
            self.left_inner_line[1],
            self.right_court_line[1],
        ],
        4: [
            self.left_court_line[0],
            self.right_inner_line[0],
            self.left_court_line[1],
            self.right_inner_line[1],
        ],
        5: [*self.top_inner_line, *self.bottom_inner_line],
        6: [
            *self.top_inner_line,
            self.left_inner_line[1],
            self.right_inner_line[1],
        ],
        7: [
            *self.bottom_inner_line,
            self.left_inner_line[1],
            self.right_inner_line[1],
        ],
        8: [
            self.right_inner_line[0],
            self.right_court_line[0],
            self.right_inner_line[1],
            self.right_court_line[1],
        ],
        9: [
            self.left_court_line[0],
            self.left_inner_line[0],
            self.left_court_line[1],
            self.left_inner_line[1],
        ],
        10: [
            self.top_inner_line[0],
            self.middle_line[0],
            self.bottom_inner_line[0],
            self.middle_line[1],
        ],
        11: [
            self.middle_line[0],
            self.top_inner_line[1],
            self.middle_line[1],
            self.bottom_inner_line[1],
        ],
        12: [
            *self.bottom_inner_line,
            self.left_inner_line[1],
            self.right_inner_line[1],
        ],
    }
    self.line_width = 1
    self.court_width = 1117
    self.court_height = 2408
    self.top_bottom_border = 549
    self.right_left_border = 274
    self.court_total_width = self.court_width + self.right_left_border * 2
    self.court_total_height = self.court_height + self.top_bottom_border * 2
    self.court = self.build_court_reference()

    # self.court = cv2.cvtColor(cv2.imread('court_configurations/court_reference.png'), cv2.COLOR_BGR2GRAY)


  def build_court_reference(self):
    """
    Create court reference image using the lines positions
    """
    court = np.zeros(
        (
            self.court_height + 2 * self.top_bottom_border,
            self.court_width + 2 * self.right_left_border,
        ),
        dtype=np.uint8,
    )
    cv2.line(court, *self.baseline_top, 1, self.line_width)
    cv2.line(court, *self.baseline_bottom, 1, self.line_width)
    cv2.line(court, *self.net, 1, self.line_width)
    cv2.line(court, *self.top_inner_line, 1, self.line_width)
    cv2.line(court, *self.bottom_inner_line, 1, self.line_width)
    cv2.line(court, *self.left_court_line, 1, self.line_width)
    cv2.line(court, *self.right_court_line, 1, self.line_width)
    cv2.line(court, *self.left_inner_line, 1, self.line_width)
    cv2.line(court, *self.right_inner_line, 1, self.line_width)
    cv2.line(court, *self.middle_line, 1, self.line_width)
    court = cv2.dilate(court, np.ones((5, 5), dtype=np.uint8))

    return court


