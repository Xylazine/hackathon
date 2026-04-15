import kagglehub
import pandas as pd
import os
from skimage import measure, filters, color, io


cache_path = os.path.expanduser(
    "~/.cache/kagglehub/datasets/iarunava/cell-images-for-detecting-malaria"
)

if not os.path.exists(cache_path):
    print("Downloading dataset...")
    path = kagglehub.dataset_download("iarunava/cell-images-for-detecting-malaria")
else:
    print("Dataset already cached, skipping download.")
    path = cache_path

print(f"Dataset path: {path}")

image_path = path + "/versions/1/cell_images/"
print(os.path.exists(image_path + "Parasitized"))
print(path)


ind = 0
rows = []

for label, folder in [('parasitized', "Parasitized"), ('uninfected', "Uninfected")]:
    ind = 0
    for filename in os.listdir(image_path + folder):
        if ind > 5:
            break
        print(filename)
        image = io.imread(image_path + folder + "/" + filename)
        print("image imported")
        # Convert to grayscale
        gray = color.rgb2gray(image)
        # Threshold to isolate cell
        thresh = filters.threshold_otsu(gray)
        binary = gray > thresh
        # Measure properties
        labeled = measure.label(binary)
        props = measure.regionprops(labeled)[0]
        rows.append({
            'source_image': filename,
            'area': props.area,
            'perimeter': props.perimeter,
            'eccentricity': props.eccentricity,
            'mean_intensity': gray.mean(),
            'label': label
        })
        ind += 1

df = pd.DataFrame(rows)
print(df)
df.describe()
df.to_csv('features.csv', index=False)


