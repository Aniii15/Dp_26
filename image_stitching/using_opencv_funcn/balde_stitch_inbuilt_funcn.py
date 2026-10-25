import cv2
import glob

# 1. Find all jpeg images in your data folder
# (Change the path to wherever your images are stored)
image_paths = glob.glob("scan_20261021_214947/*.*")
images_to_merge = []

# 2. Read every image and add it to the list
for path in image_paths:
    img = cv2.imread(path)
    if img is not None:
        images_to_merge.append(img)

print(f"Found {len(images_to_merge)} images to stitch.")

# 3. Create the stitcher in SCANS mode (to prevent 3D squishing)
stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)

# 4. Feed the entire dataset into the stitcher at once
status, stitched_image = stitcher.stitch(images_to_merge)

if status == cv2.Stitcher_OK:
    cv2.imwrite("massive_combined_image.jpeg", stitched_image)
    print("Success! All images merged into one.")
else:
    print(f"Stitching failed with error code: {status}")
