"""
This image_data_drift.py is only applied in the local test
"""

import numpy as np
import pandas as pd
from torchvision import transforms
import os
from PIL import Image
from google.cloud import storage
import json
from evidently.metrics import DataDriftTable
from evidently.report import Report


# Feature extraction function
def extract_features(image):
    """Extract basic image features from a single image."""
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),  # Resize image to 128x128
            transforms.ToTensor(),  # Convert to tensor (CxHxW format)
        ]
    )
    image_tensor = transform(image)
    input_numpy = image_tensor.numpy()  # Convert to NumPy
    avg_brightness = np.mean(input_numpy)
    contrast = np.std(input_numpy)
    sharpness = np.mean(np.abs(np.gradient(input_numpy)))
    return [avg_brightness, contrast, sharpness]


# Define image loading function
def load_images_and_extract_features(folder_path, label):
    """Load images from a folder, extract features, and return a list of features with labels."""
    features_list = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        try:
            # Open image and convert to RGB
            with Image.open(file_path).convert("RGB") as img:
                features = extract_features(img)
                features_list.append(features + [label])
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    return features_list


# Define paths
hotdog_path = "data/train/hotdog"
nothotdog_path = "data/train/nothotdog"

# Extract features
hotdog_features = load_images_and_extract_features(hotdog_path, "hotdog")
nothotdog_features = load_images_and_extract_features(nothotdog_path, "nothotdog")

# Combine features
all_features = hotdog_features + nothotdog_features

# Create DataFrame
columns = ["avg_brightness", "contrast", "sharpness", "category"]
reference_df = pd.DataFrame(all_features, columns=columns)

# Save as CSV file
reference_df.to_csv("reference_data.csv", index=False)
print("Feature extraction completed. Saved to 'reference_data.csv'.")


# Set Google Cloud credentials
key_file = os.path.join("./keys/cloud_storage_key.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_file

# Initialize storage client
storage_client = storage.Client()
bucket = storage_client.bucket("mlops-trained-models")

# Retrieve all JSON files from the storage bucket
blobs = bucket.list_blobs()
json_files = [blob for blob in blobs if blob.name.endswith(".json")]

# Merge JSON file contents
dataframes = []
for json_blob in json_files:
    temp_file = json_blob.name.replace("/", "_")  # Temporary file name
    json_blob.download_to_filename(temp_file)  # Download JSON file
    try:
        # Load JSON file
        with open(temp_file, "r") as f:
            data = json.load(f)

        # Construct DataFrame based on JSON structure
        if isinstance(data, list):  # If the JSON is a list
            df = pd.DataFrame(data)
        elif isinstance(data, dict):  # If the JSON is a dictionary
            df = pd.DataFrame([data])  # Convert to single-row DataFrame
        else:
            print(f"Unsupported JSON format in {json_blob.name}")
            continue

        dataframes.append(df)
    except Exception as e:
        print(f"Error reading {json_blob.name}: {e}")
    finally:
        os.remove(temp_file)  # Delete temporary file

# Combine all DataFrames
if dataframes:
    current_df = pd.concat(dataframes, ignore_index=True)
    current_df.to_csv("current_data.csv", index=False)
    print("All JSON files have been merged into 'current_data.csv'.")
else:
    print("No valid JSON files found in the bucket.")


"""
Below is the code to compare the reference and current data and generate a report on the data drift.
"""
reference_df["category"] = reference_df["category"].apply(lambda x: 1 if x == "hotdog" else 0)
current_df["category"] = current_df["category"].apply(lambda x: 1 if x == "hotdog" else 0)

# Drop non-numerical columns
numerical_columns = ["avg_brightness", "contrast", "sharpness", "category"]
reference_features = reference_df[numerical_columns]
current_features = current_df[numerical_columns]

# Initialize Evidently report for data drift
report = Report(metrics=[DataDriftTable()])

# Run the report with reference and current data
report.run(reference_data=reference_features, current_data=current_features)

# Save the report as an HTML file
report.save_html("data_drift_report.html")
print("Data drift report has been saved to 'data_drift_report.html'.")
