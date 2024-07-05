import argparse
import logging
import pathlib
import functools

import cv2
import torch
from torchvision import transforms

from semantic_segmentation import models
from semantic_segmentation import load_model
from semantic_segmentation import draw_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)

    parser.add_argument("--model-type", type=str, choices=models, required=True)

    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--save", action="store_true")
    parser.add_argument("--display", action="store_true")

    return parser.parse_args()


def find_files(dir_path: pathlib.Path, file_exts):
    assert dir_path.exists()
    assert dir_path.is_dir()

    for file_ext in file_exts:
        yield from dir_path.rglob(f"*{file_ext}")


def _load_image(image_path: pathlib.Path):
    image = cv2.imread(str(image_path))
    assert image is not None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_width = (image.shape[1] // 32) * 32
    image_height = (image.shape[0] // 32) * 32

    image = image[:image_height, :image_width]
    return image



import numpy as np

# def average_rgb(mask, image):
#     # Ensure mask is broadcastable to the image
#     if mask.shape[2:] != image.shape[:2]:
#         raise ValueError("Mask and image dimensions do not match.")

#     # Apply the mask to each channel
#     masked_image = np.multiply(image, mask.transpose(1, 2, 3, 0))  # Broadcasting mask to image dimensions

#     # Compute the sum of the masked pixels for each channel
#     sum_rgb = np.sum(masked_image, axis=(0, 1))

#     # Compute the number of masked pixels
#     num_masked_pixels = np.sum(mask)

#     # Compute the average for each channel
#     avg_rgb = sum_rgb / num_masked_pixels

#     return avg_rgb


def average_rgb(mask, image):
    # Convert the mask to a NumPy array and ensure it is broadcastable to the image
    mask_np = mask.squeeze().cpu().numpy()
    
    # Apply the mask to each channel
    masked_image = image * mask_np[:, :, np.newaxis]
    
    # Compute the sum of the masked pixels for each channel
    sum_rgb = np.sum(masked_image, axis=(0, 1))
    
    # Compute the number of masked pixels
    num_masked_pixels = np.sum(mask_np)
    
    # Compute the average for each channel
    avg_rgb = sum_rgb / num_masked_pixels
    
    return avg_rgb

    

import cv2
import numpy as np
from sklearn.metrics import pairwise_distances_argmin

# Define Monk Skin Tone (MST) values
mst_values = [
    (246, 237, 228), # Lightest
    (243, 231, 219),
    (247, 234, 208),
    (234, 218, 186),
    (215, 189, 150),
    (160, 126, 86),
    (130, 92, 67),
    (96, 65, 52),
    (58, 49, 42),
    (41, 36, 32), # Darkest
]


def classify_skin_tone(avg_skin_color):

    # Find the closest MST value
    # print(avg_skin_color)
    closest_mst_index = pairwise_distances_argmin([avg_skin_color], mst_values)[0] + 1  # start from 1

    return closest_mst_index, mst_values[closest_mst_index]


if __name__ == "__main__":
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    assert args.display or args.save

    model = torch.load(args.model, map_location=device)
    print(models)
    print(args.model_type)
    model = load_model(models[args.model_type], model)
    model.to(device).eval()
    print('djsaldjas')

    image_dir = pathlib.Path(args.images)

    fn_image_transform = transforms.Compose(
        [
            transforms.Lambda(lambda image_path: _load_image(image_path)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    for image_file in find_files(image_dir, [".png", ".jpg", ".jpeg"]):
        print(image_file)
        print('djasldsa')
        image = fn_image_transform(image_file)

        with torch.no_grad():
            image = image.to(device).unsqueeze(0)
            results = model(image)["out"]
            results = torch.sigmoid(results)

            results = results > args.threshold

            # print(results)
            # print(results.size())

        # print(model.categories)
        for category, category_image, mask_image in draw_results(image[0], results[0], categories=model.categories):
            if args.save:
                output_name = f"results_{category}_{image_file.name}"
                cv2.imwrite(str(output_name), category_image)
                cv2.imwrite(f"mask_{category}_{image_file.name}", mask_image)
                # print(type(mask_image))
                # print(mask_image.shape)
                # print(mask_image)

                avg_rgb = average_rgb(results, mask_image)

                # print(avg_rgb)

                skin_tone_index, skin_tone_value = classify_skin_tone(avg_rgb)
                # print(image_file.name)
                print(f"The closest Monk Skin Tone value is: {skin_tone_value} (index {skin_tone_index}) \n")

