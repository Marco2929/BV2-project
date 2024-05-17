from PIL import Image
import os

# Function to convert a PPM file to a PNG file
def convert_ppm_to_png(ppm_file_path):
    # Open the PPM file
    with Image.open(ppm_file_path) as img:
        # Get the base name of the file (without extension)
        base_name = os.path.splitext(ppm_file_path)[0]
        # Create the PNG file path by adding .png extension
        png_file_path = f"{base_name}.png"
        # Save the image as PNG
        img.save(png_file_path, 'PNG')
        print(f"Converted {ppm_file_path} to {png_file_path}")

# Function to delete a PPM file
def delete_ppm_file(ppm_file_path):
    try:
        os.remove(ppm_file_path)
        print(f"Deleted {ppm_file_path}")
    except Exception as e:
        print(f"Error deleting {ppm_file_path}: {e}")

if __name__ == "__main__":
    # Directory containing PPM files
    ppm_directory = r'C:\Users\Benedikt Seeger\Downloads\walk\TrainIJCNN2013\TrainIJCNN2013'

    # Convert all PPM files in the directory
    for file_name in os.listdir(ppm_directory):
        if file_name.endswith('.ppm'):
            ppm_file_path = os.path.join(ppm_directory, file_name)
            convert_ppm_to_png(ppm_file_path)
