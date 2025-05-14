import zipfile
import os

zip_file = "C:/Users/bright/Downloads/newdata.zip" 

# Set the extraction path
extract_to = "C:/Users/bright/Documents/sea_ice_data"  # Change this as needed

# Create the extraction directory if it doesn't exist
os.makedirs(extract_to, exist_ok=True)

# Extract the ZIP file
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

# List the files in the extracted directory
extracted_files = os.listdir(extract_to)
print("Extracted files:", extracted_files)
