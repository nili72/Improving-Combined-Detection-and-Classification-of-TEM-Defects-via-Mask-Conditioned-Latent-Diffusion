import json
from PIL import Image, ImageDraw, ImagePath
import os
import numpy as np
import cv2
import shutil
import pandas as pd
from pathlib import Path
import random

# Parameters
option = 'train'  # 'train' or 'val'
imganno_save = False
size = 256
area_threshold = 0
sample_count =10

skip_image = 0
success_count = 0

# Set paths
# current_dir = Path.cwd()

parent_directory = Path('/ocean/projects/mat240020p/nli1/diffusion/data/dataset2/masks/train/split4/100/256_crop_5_rot_flip/')
# parent_directory = Path('/Users/nuohaoliu/Documents/data_local/val/tiny')


mask_directory = parent_directory

# Ensure necessary directories exist
create_folder = [
    parent_directory / 'annotations',
    # parent_directory /  f'{option}',
    
]
for folder in create_folder:
    folder_path = Path(folder)
    folder_path.mkdir(parents=False, exist_ok=True)

# annotation_path = parent_directory / 'annotations' / 'val.json'
if imganno_save:
    image_directory = parent_directory / 'train'
    image_directory / f'image_anno_f{area_threshold}'
    create_folder = [
        # parent_directory / 'annotations',
        parent_directory /  f'{option}',
        
    ]
    for folder in create_folder:
        folder_path = Path(folder)
        folder_path.mkdir(parents=False, exist_ok=True)
    
    imageanno_directory = image_directory  / f'image_anno_f{area_threshold}'
# output_directory = parent_directory / 'crop' / f'{option}'
output_annotation_path = parent_directory /  'annotations' / f'{option}.json'

# output_directory.mkdir(parents=True, exist_ok=True)



 
train_dict = dict()
train_dict["licenses"] = [
    {
      "id": 1,
      "name": "",
      "url": ""
    }
  ]
train_dict["categories"]= [
    {
      "id": 1,
      "name": "np",
      "supercategory": "np"
    }
  ]
train_dict["info"]= {
    "contributor": "",
    "date_created": "",
    "description": "New dataset",
    "url": "",
    "version": "1.0",
    "year": 2024
  },
train_dict["images"] = list()
train_dict["annotations"] = list()
 
defect_count = 0
image_count = 0
# gray_levels = [85, 170, 255]
# Define gray levels and their ranges
gray_level_ranges = {
    85: (75, 96),  # Example: 85 ± 5
    170: (160, 180),  # Example: 170 ± 5
    255: (245, 270)  # Example: 255 ± 5
}


# Define a list of valid image extensions
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

# Get the list of image names in the directory with valid extensions
train_img_list = [
    image_name for image_name in os.listdir(mask_directory)
    if image_name.lower().endswith(valid_extensions)
]

def process_mask(im_name):
    # Load the mask and original image
    im_arr = cv2.imread(str(mask_directory / im_name), cv2.IMREAD_GRAYSCALE)

    y_size_mask = im_arr.shape[0]
    x_size_mask = im_arr.shape[1]
    # print('Mask size', x_size_mask, y_size_mask)

    # Check if size matches the required dimensions
    if not y_size_mask == x_size_mask == size:
        raise ValueError("Mask size doesn't match expected size")

    # Prepare for multiple class categories based on gray levels
    # gray_levels = [85, 170, 255]  # Example gray levels for different classes
    class_annotations = {level: {"bboxes": [], "polygons": []} for level in gray_level_ranges}

    # Set the area threshold
    # area_threshold = 500  # Change this value as needed

    for gray_level, (lower_bound, upper_bound) in gray_level_ranges.items():
        # Isolate mask region for the current class within the range
        class_mask = cv2.inRange(im_arr, lower_bound, upper_bound)
        # Isolate mask region for the current class
        # class_mask = (im_arr == gray_level).astype(np.uint8) * 255

        # Get contours for the current class
        contours, _ = cv2.findContours(class_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Filter out small contours based on area
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= area_threshold]

        # Extract polygons and bounding boxes for the current class
        for obj in filtered_contours:
            coords = []
            for point in obj:
                coords.append(int(point[0][0]))
                coords.append(int(point[0][1]))

            # Only append polygons that have 3 or more points
            if len(coords) >= 6:
                # Add the polygon to the current class
                class_annotations[gray_level]["polygons"].append(coords)

                # Calculate bounding box
                xs = coords[::2]  # Even indices are x-coordinates
                ys = coords[1::2]  # Odd indices are y-coordinates
                minx, maxx = min(xs), max(xs)
                miny, maxy = min(ys), max(ys)
                class_annotations[gray_level]["bboxes"].append([minx, miny, maxx, maxy])

    return class_annotations, x_size_mask, y_size_mask

def process_image(im_name):
    # Load the mask and original image
    im_arr = cv2.imread(str(mask_directory / im_name), cv2.IMREAD_GRAYSCALE)
    im_raw_arr = cv2.imread(str(image_directory / im_name))
    im_raw = Image.fromarray(im_raw_arr)

    # Get shape of mask and original image
    y_size = im_raw_arr.shape[0]
    x_size = im_raw_arr.shape[1]
    print('Image size', x_size, y_size)
    y_size_mask = im_arr.shape[0]
    x_size_mask = im_arr.shape[1]
    print('Mask size', x_size_mask, y_size_mask)

    # Resize the mask to match the original image size
    im_arr = cv2.resize(im_arr, (x_size, y_size))

    print('Resized', im_arr.shape[1], im_arr.shape[0])

    # Prepare for multiple class categories based on gray levels
    gray_levels = {
    85: (80, 90),  # Example: 85 ± 5
    170: (165, 175),  # Example: 170 ± 5
    255: (250, 260)  # Example: 255 ± 5
}
    class_annotations = {level: {"bboxes": [], "polygons": []} for level in gray_levels}
    
    # Set the area threshold
    # area_threshold = 500  # Change this value as needed

    for gray_level, (lower_bound, upper_bound) in gray_level_ranges.items():
        # Isolate mask region for the current class within the range
        class_mask = cv2.inRange(im_arr, lower_bound, upper_bound)

        # Get contours for the current class
        contours, _ = cv2.findContours(class_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Filter out small contours based on area
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= area_threshold]

        # Extract polygons and bounding boxes for the current class
        for obj in filtered_contours:
            coords = []
            for point in obj:
                coords.append(int(point[0][0]))
                coords.append(int(point[0][1]))

            # Only append polygons that have 3 or more points
            if len(coords) >= 6:
                # Add the polygon to the current class
                class_annotations[gray_level]["polygons"].append(coords)

                # Calculate bounding box
                xs = coords[::2]  # Even indices are x-coordinates
                ys = coords[1::2]  # Odd indices are y-coordinates
                minx, maxx = min(xs), max(xs)
                miny, maxy = min(ys), max(ys)
                class_annotations[gray_level]["bboxes"].append([minx, miny, maxx, maxy])


    # Plot annotations on the image for visualization
    im1 = ImageDraw.Draw(im_raw)
    for gray_level, data in class_annotations.items():
        color = "blue" if gray_level == 85 else "green" if gray_level == 170 else "red"
        for polygon in data["polygons"]:
            im1.polygon(polygon, outline=color)

        for bbox in data["bboxes"]:
            im1.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])), outline=color)

    return class_annotations, im_raw, x_size, y_size


# Define the fraction or count of images to sample
  # Adjust to your preference
sampled_images = random.sample(train_img_list, int(min(len(train_img_list), sample_count)))


# Choose which set of images to iterate over
if imganno_save:
    # 1) Loop over ALL images to add minimal info to train_dict
#    (e.g., store mask info or at least a placeholder).
#    You can call `process_mask(im_name)` if you want to store the mask polygons for each image.

    image_count = 0
    for i, im_name in enumerate(train_img_list, start=1):
        try:
            # If you want to extract basic mask info for *all* images, call `process_mask`:
            
            class_annotations, x_size, y_size = process_mask(im_name)
            # print('Processing image:', im_name)
            print(f"Processing image {i}/{len(train_img_list)}: {im_name}")
            # You can store the mask-based bboxes/polygons in a top-level data structure
            # or simply store the image metadata here. 
            # For instance, add an 'images' entry to train_dict:
            d = {
                "coco_url": "",
                "date_captured": "",
                "file_name": im_name,
                "flickr_url": "",
                "height": y_size,
                "id": image_count,
                "license": 1,
                "width": x_size
            }
            train_dict['images'].append(d)

            # Append annotation data for only the sampled images
            for gray_level, data in class_annotations.items():
                for bbox, polygon in zip(data["bboxes"], data["polygons"]):
                    bbox_mask = [
                        bbox[0], 
                        bbox[1], 
                        bbox[2] - bbox[0],  # width
                        bbox[3] - bbox[1]   # height
                    ]

                    # Map the range-based gray_level to category_id
                    category_id = None
                    for idx, (level, (lower_bound, upper_bound)) in enumerate(gray_level_ranges.items()):
                        if lower_bound <= gray_level <= upper_bound:
                            category_id = idx + 1  # Category IDs are 1-based
                            break
                    if category_id is None:
                        # If no category_id found, skip
                        continue

                    d = {
                        "image_id": image_count,
                        "id": defect_count,
                        "category_id": category_id,
                        "segmentation": [polygon],
                        "bbox": bbox_mask,
                        "area": bbox_mask[2] * bbox_mask[3],
                        "iscrowd": 0,
                    }
                    train_dict['annotations'].append(d)
                    defect_count += 1
            
            # Optionally, you can store preliminary polygons here if you want
            # to keep "mask info" for all images. For example:
            # (But if you only want to do annotation for sampled images, you might skip writing them here.)
            
            # for gray_level, data in class_annotations.items():
            #    # Store or do something with data['bboxes'] / data['polygons']
            #    # (No final annotation yet, unless you actually want partial annotation.)
            
            image_count += 1

        except Exception as e:
            print(f"Error processing mask for {im_name}: {e}")
            skip_image += 1
            continue

    print(f"Added base info for {image_count} images into train_dict")
    
    # 2) Now handle SAMPLED images for full annotation & saving
    #    This is where we call `process_image(...)` and store polygons, bounding boxes, etc.

    # If you still need the dictionary that maps a file name -> image ID (from the first step),
    # create a quick lookup. For example:
    filename_to_imgid = {}
    for img_info in train_dict['images']:
        filename_to_imgid[img_info["file_name"]] = img_info["id"]

    # Loop over only the sampled subset
    for im_name in sampled_images:
        print('Processing annotation for sampled image:', im_name)
        try:
            class_annotations, im_raw, x_size, y_size = process_image(im_name)
            success_count += 1
        except Exception as e:
            print(f"Error processing image {im_name}: {e}")
            skip_image += 1
            continue

        # Save the annotated image
        im_raw.save(imageanno_directory / im_name)
        # print("saved annotated image:", im_name, x_size, y_size)

        # Retrieve the 'id' from step (1). If you didn't store it, you can re-assign:
        image_id = filename_to_imgid.get(im_name)
        if image_id is None:
            # If for some reason it wasn't added in step (1), handle gracefully
            # e.g., skip or create a new entry
            continue

    print(f"Annotated and saved {success_count} sampled images, skipped {skip_image} images")


        
else:
    for i, im_name in enumerate(train_img_list):
        try:
            class_annotations,x_size, y_size = process_mask(im_name)
            success_count += 1
            print(f"Processing image {i}/{len(train_img_list)}: {im_name}")
        except Exception as e:
            print(f"Error processing image {im_name}: {e}")
            # Optionally handle the error, e.g., skip this image or log the issue
            skip_image += 1
            continue
        
        
    
    
    # shutil.copy(image_directory + im_name, save_im+'/')
 
    # # Save data for YOLO
    # with open(save_anno+'/'+im_name[0:-5]+'.txt', 'w') as f:
    #     for bbox_frac in bboxes_frac:
    #         f.write('0 '+str(bbox_frac[0])+' '+str(bbox_frac[1])+' '+str(bbox_frac[2])+' '+str(bbox_frac[3])+'\n')
 
    # Append data for CoCo (Mask)

        d =  {"coco_url": "",
        "date_captured": "",
        "file_name": im_name,
        "flickr_url": "",
        "height": y_size,
        "id": image_count,
        "license": 1,
        "width": x_size}
        train_dict['images'].append(d)
    
        for gray_level, data in class_annotations.items():
                for bbox, polygon in zip(data["bboxes"], data["polygons"]):
                    bbox_mask = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
                    
                    category_id = None
                    for idx, (level, (lower_bound, upper_bound)) in enumerate(gray_level_ranges.items()):
                        if lower_bound <= gray_level <= upper_bound:
                            category_id = idx + 1  # Category IDs are 1-based
                            break
                    if category_id is None:
                        continue
                    d = {
                        "image_id": image_count,
                        "id": defect_count,
                        "category_id": category_id,  # Map gray level to category_id
                        "segmentation": [polygon],
                        "bbox": bbox_mask,
                        "area": bbox_mask[2] * bbox_mask[3],
                        "iscrowd": 0,
                    }
                    train_dict['annotations'].append(d)
                    defect_count += 1
    
        image_count += 1
 
with open(output_annotation_path, 'w') as f:
    json.dump(train_dict, f, indent=4)
print(f"process {image_count} images, skip {skip_image} images")
print(f"Final train_dict has {len(train_dict['images'])} images, "
      f"{len(train_dict['annotations'])} annotations")