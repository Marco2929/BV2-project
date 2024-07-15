import os


def change_yolo_classes(directory, new_class):
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        # Check if the file is a .txt file
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)

            # Read the content of the file
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Change the class id to the new class
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    parts[0] = str(new_class)
                    new_line = ' '.join(parts)
                    new_lines.append(new_line)

            # Write the updated content back to the file
            with open(file_path, 'w') as file:
                file.write('\n'.join(new_lines))

if __name__ == '__main__':
    # Example usage:
    directory_path = r'C:\Users\Benedikt Seeger\PycharmProjects\BV2-project\tools\coco_tools\coco_selected_annotations_traffic_light'
    new_class_id = 9
    change_yolo_classes(directory_path, new_class_id)
