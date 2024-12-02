import os
import cv2
import csv
import numpy as np
from skimage import measure
import joblib
from tensorflow.keras.models import load_model



MODEL_PATH = 'model/SegModels'
IMAGE_PATH = 'uploads'
target_size = (512, 512) 

def segment_image(original_image_path, mask_image_path):
    '''
    This function segments the original image using the mask image and saves the segmented image.

    Args:
        original_image_path (str): The path to the original image.
        mask_image_path (str): The path to the mask image.
        output_image_path (str): The path to save the segmented image.

    Returns:
        bool: True if successful, False otherwise.
    '''
    original_image = cv2.imread(original_image_path)
    original_image = cv2.resize(original_image, (512, 512))
    mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
    mask_image = cv2.resize(mask_image, (512, 512))

    if original_image is None or mask_image is None:
        print(f"Error reading image or mask for {original_image_path}")
        return False
    
    # Make sure the mask is binary
    _, binary_mask = cv2.threshold(mask_image, 0, 255, cv2.THRESH_BINARY)

    segmented_image = cv2.bitwise_and(original_image, original_image, mask=binary_mask)
    return segmented_image
    # cv2.imwrite(output_image_path, segmented_image)
    # return True


def predict_segmentation(model, preprocessed_image):
    prediction = model.predict(preprocessed_image)
    prediction = np.squeeze(prediction, axis=0) 
    prediction = (prediction > 0.5).astype(np.uint8)  
    return prediction


def calculate_average_rgb(image):
    '''
    This function calculates the average RGB values of an image.

    Args:
        image_path (str): The path to the image.

    Returns:
        numpy.ndarray: The average RGB values as a NumPy array.
    '''

    average_rgb = np.mean(image, axis=(0, 1))
    return average_rgb


def blur_image(image):
    '''
    This function blurs an image using a Gaussian blur and saves the blurred image.

    Args:

    Returns:
        bool: True if successful, False otherwise.
    '''
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    labels = measure.label(binary, connectivity=2)

    properties = measure.regionprops(labels)
    largest_region = max(properties, key=lambda x: x.area)
    
    return largest_region

def preprocess_image(image_path, target_size):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image / 255.0 
    image = np.expand_dims(image, axis=0) 
    return image




def image_to_mean_rgb(file_names, body_parts):
    '''['tongue', 'right_fingernail', 'left_fingernail', 'left_palm', 'right_palm', 'left_eye', 'right_eye']'''
    average_rgb_values = []
    results = {}
    for i in range(7):
        filename = file_names[i]
        body_part = body_parts[i]
        model_name = ''
        print(body_part)
        # Load the appropriate model
        if body_part.endswith("eye"):
            model_name = 'eyelid_model.pkl'
        elif body_part.endswith("fingernail"):
            model_name = 'fingernail_model.pkl'
        elif body_part.endswith("palm"):
            model_name = 'palm_model.pkl'
        else:
            continue

        # Load image and model
        image_path = os.path.join(IMAGE_PATH, filename)
        model_path = os.path.join(MODEL_PATH, model_name)
        print(model_name)

        model = load_model(model_path)
        preprocessed_image = preprocess_image(image_path, target_size)
        prediction = predict_segmentation(model, preprocessed_image)


        # If the body part is not a fingernail, blur the image using gaussian blur
        if not body_part.endswith("fingernail"):
            image = prediction * 255
            blurred = cv2.GaussianBlur(image, (5, 5), 0)
            _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
            labels = measure.label(binary, connectivity=2)
            properties = measure.regionprops(labels)
            if len(properties) == 0:
                print("No regions found")
                largest_region_mask = np.zeros_like(binary)
            else:
                largest_region = max(properties, key=lambda x: x.area)
                largest_region_mask = np.zeros_like(binary)
                largest_region_mask[labels == largest_region.label] = 255
            mask = largest_region_mask
        else:
            mask = prediction * 255
        

        # Segment the image
        segmented_image = segment_image(image_path, mask)

        # calculate average rgb
        average_rgb = calculate_average_rgb(segmented_image)
        average_rgb_values.append(average_rgb)
        results[body_part] = average_rgb
    return results


#if "__name__" == "__main__":
values = image_to_mean_rgb(['tongue.jpg', 'right_nail.jpg', 'left_nail.jpg', 'left_palm.jpg', 'right_palm.jpg', 'left_eye.jpg', 'right_eye.jpg'], 
                    ['tongue', 'right_fingernail', 'left_fingernail', 'left_palm', 'right_palm', 'left_eye', 'right_eye'])
print(values)
print("Done")