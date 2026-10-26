import cv2
import glob

# Initialize the ORB detector to quickly find features
orb = cv2.ORB_create()
# Minimum number of features required to keep the image (you can tweak this)
MIN_FEATURES = 30

# 1. Find all jpeg images in your data folder
image_paths = glob.glob("test_images/*.*")
images_to_merge = []

# 2. Read every image, check for features, and add it to the list
for path in image_paths:
    img = cv2.imread(path)
    if img is not None:
        # Convert to grayscale just for the feature check (it's faster)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect keypoints
        keypoints, _ = orb.detectAndCompute(gray, None)

        # Only append if the image has enough distinct features
        if len(keypoints) >= MIN_FEATURES:
            images_to_merge.append(img)
        else:
            print(f"Skipping {path}: Only found {len(keypoints)} features.")

print(
    f"Filtered down to {len(images_to_merge)} good images out of {len(image_paths)} total.")

# 3. Create the stitcher in SCANS mode
stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)

# 4. Feed the filtered dataset into the stitcher at once
status, stitched_image = stitcher.stitch(images_to_merge)

if status == cv2.Stitcher_OK:
    cv2.imwrite("massive_combined_image2.jpeg", stitched_image)
    print("Success! All images merged into one.")
else:
    print(f"Stitching failed with error code: {status}")
