import os


def create_txt_for_png(directory):
    """
    This function checks for .png files in the specified directory.
    For each .png file, it creates a .txt file with the same name if it does not already exist.

    :param directory: Path to the directory to check for .png files
    """
    # Check if the directory exists
    if not os.path.exists(directory):
        print(f"The directory {directory} does not exist.")
        return

    # Get a list of all files in the directory
    files = os.listdir(directory)

    # Process each .png file
    for file in files:
        if file.lower().endswith('.png'):
            # Generate the corresponding .txt filename
            txt_file = file[:-4] + '.txt'
            txt_file_path = os.path.join(directory, txt_file)

            # Check if the .txt file exists, if not, create it
            if not os.path.exists(txt_file_path):
                open(txt_file_path, 'w').close()
                print(f"Created an empty file: {txt_file_path}")
            else:
                print(f"The file {txt_file_path} already exists.")

    print("All .png files have a .txt counterpart.")


# Example usage
directory_path = r'C:\Users\Benedikt Seeger\Downloads\gtsdb'
create_txt_for_png(directory_path)

