defaultkey = "default"

# MSEE Trigger coordinates
CHECK_OPEN_COORDS_ONE = [[1471, 67], [1487, 117]]
TRIGGER_ROI_COORDS_ONE = [[1348, 72], [1640, 671]]

CHECK_OPEN_COORDS_TWO = [[354, 70], [375, 110]]
TRIGGER_ROI_COORDS_TWO = [[114, 64], [600, 722]]

# Thresholds to determine whether the door is closed or not
# The higher the score - the more similar it is to the reference (close threshold because door will be most similar to closed door reference image)
# The lower the score - the less similar it is to the reference (open threshold because the an open door is least similar to a closed door reference image)
# This is w.r.t Structured similarity index. These values are derived empirically.
DOOR_CLOSED_THRESHOLD = 0.27
DOOR_OPEN_THRESHOLD = 0.87
