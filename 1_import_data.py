import kagglehub
import pandas as pd
import os
from skimage import measure, filters, color


# Only download if the dataset doesn't already exist
if not os.path.exists("/cell-images-for-detecting-malaria"):
    print("Downloading dataset...")
    path = kagglehub.dataset_download("iarunava/cell-images-for-detecting-malaria")
else:
    print("Dataset already exists, skipping download.")

rows = []
for label, folder in [('parasitized', "Parasitized"), ('uninfected', "Uninfected")]:
    print(folder)
    #for filename in os.listdir(folder):
        # image = load_image(filename)
        # # Convert to grayscale
        # gray = color.rgb2gray(image)
        # # Threshold to isolate cell
        # thresh = filters.threshold_otsu(gray)
        # binary = gray > thresh
        # # Measure properties
        # props = measure.regionprops(binary)[0]
        # rows.append({
        #     'source_image': filename,
        #     'area': props.area,
        #     'perimeter': props.perimeter,
        #     'eccentricity': props.eccentricity,
        #     'mean_intensity': gray.mean(),
        #     'label': label
        # })

# df = pd.DataFrame(rows)
# df.to_csv('features.csv', index=False)


