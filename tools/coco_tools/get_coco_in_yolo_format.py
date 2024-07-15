import os
import requests
from pycocotools.coco import COCO
import zipfile

# Function to download and extract COCO dataset
def download_and_extract_coco(dataset_url, dataset_dir):
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    dataset_zip_path = os.path.join(dataset_dir, 'coco.zip')
    with requests.get(dataset_url, stream=True) as r:
        with open(dataset_zip_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)
    os.remove(dataset_zip_path)

# Function to download images and save bounding boxes in YOLO format
def download_coco_images_and_annotations(coco, image_ids, save_dir, annotations_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(annotations_dir):
        os.makedirs(annotations_dir)
    for img_id in image_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_url = img_info['coco_url']
        img_path = os.path.join(save_dir, img_info['file_name'])
        if not os.path.exists(img_path):
            with requests.get(img_url, stream=True) as r:
                with open(img_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        # Save annotations in YOLO format
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=relevant_class_ids, iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        annotation_path = os.path.join(annotations_dir, f"{os.path.splitext(img_info['file_name'])[0]}.txt")
        with open(annotation_path, 'w') as f:
            for ann in anns:
                bbox = ann['bbox']
                # Convert COCO bbox (x, y, width, height) to YOLO format (x_center, y_center, width, height)
                x_center = (bbox[0] + bbox[2] / 2) / img_info['width']
                y_center = (bbox[1] + bbox[3] / 2) / img_info['height']
                width = bbox[2] / img_info['width']
                height = bbox[3] / img_info['height']
                yolo_class = coco_to_yolo_class[ann['category_id']]
                f.write(f"{yolo_class} {x_center} {y_center} {width} {height}\n")

if __name__ == "__main__":
    # Download annotations if not present
    annotations_dir = 'coco/annotations'
    if not os.path.exists(annotations_dir):
        download_and_extract_coco(
            'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
            'coco'
        )

    # Load COCO annotations
    coco = COCO(os.path.join(annotations_dir, 'instances_train2017.json'))

    # Define relevant classes
    relevant_classes = ['person']#'car', 'person', 'traffic light', 'stop sign']
    relevant_class_ids = coco.getCatIds(catNms=relevant_classes)

    # Create a mapping from COCO class id to YOLO class id
    coco_to_yolo_class = {id: i for i, id in enumerate(relevant_class_ids)}

    # Get image ids that contain the relevant classes
    image_ids = coco.getImgIds(catIds=relevant_class_ids)

    # Select a subset of 500 image ids
    selected_image_ids = image_ids[:500]

    # Download images if not present
    images_dir = 'coco/train2017'
    if not os.path.exists(images_dir):
        download_and_extract_coco(
            'http://images.cocodataset.org/zips/train2017.zip',
            'coco'
        )

    # Download the selected images and annotations
    download_coco_images_and_annotations(coco, selected_image_ids, 'coco_selected_images_person', 'coco_selected_annotations_person')
