#USE_OLD_VERSION = False
# bookkeeping file name
STATE_ESTIMATION_BOOKKEEPING_JSON = 'state_estimation_bookkeeping.json'

# detection result json file name per image, which is saved at working directory
STATE_ESTIMATION_DATA_JSON = 'state_estimation_data.json'

# how many minutes should we consider for an incoming image to be a consecutive one
RESET_TIME_LIMIT = 10

# how much of change should be considered for the object to have "moved" tag
MOVE_THRESHOLD = 0.25 # in percentage

# bookkeeping tag macro
KEY_ADDED = "Added"
KEY_REMOVED = "Removed"
KEY_MOVED = "Moved"
KEY_POSITION = "pose"
KEY_BBOX = "part_bounding_box"
KEY_CONTOUR = "outline_poses"
KEY_PART= "part_type"
KEY_CONF= "part_type_confidence"
KEY_PART_CONF= "part_type_specifics_confidence"