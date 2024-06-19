import os

# Specify the directory containing the images and text files
directory = 'path/to/your/directory'

# List of relevant class IDs
relevant_classes = {1, 2, 9, 11}

# Iterate through all files in the directory
for filename in os.listdir(directory):
    # Process only text files
    if filename.endswith('.txt'):
        file_path = os.path.join(directory, filename)

        # Read the content of the text file
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Filter out irrelevant bounding boxes
        filtered_lines = []
        for line in lines:
            parts = line.split()
            class_id = int(parts[0])
            if class_id in relevant_classes:
                filtered_lines.append(line)

        # Write the filtered bounding boxes back to the text file
        with open(file_path, 'w') as file:
            file.writelines(filtered_lines)

print("Filtering complete.")
