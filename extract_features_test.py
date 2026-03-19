import numpy as np

features = np.load("input_features.npy")

print("Total frames:", len(features))

all_zero_frames = np.all(features == 0, axis=1)
print("All-zero frames:", np.sum(all_zero_frames))

print("Min value:", np.min(features))
print("Max value:", np.max(features))
print("Any NaNs:", np.isnan(features).any())

frame = features[0]

pose = frame[:132]
left_hand = frame[132:195]
right_hand = frame[195:258]

print("Pose nonzero count:", np.count_nonzero(pose))
print("Left hand nonzero count:", np.count_nonzero(left_hand))
print("Right hand nonzero count:", np.count_nonzero(right_hand))