import numpy as np
from PIL import Image
from moviepy.editor import VideoFileClip
from scipy.ndimage import gaussian_filter

from flask import Flask, send_file, jsonify, request

app = Flask(__name__)


# cat transformation matrix
CAT_TRANSFORMATION_MATRIX = np.array([
    [0.2, 0.5, 0.3],
    [0.2, 0.7, 0.1],
    [0.1, 0.1, 0.8]
])

# dog transformation matrix
DOG_TRANSFORMATION_MATRIX = np.array([
    [0.625, 0.375, 0.0],
    [0.700, 0.300, 0.0],
    [0.000, 0.300, 0.700]
])

# s = c log (1 + ar)
# c is an overall scaling factor
# a is a secondary scale on the original intensity
# log (1 + ar) to avoid log(0)
def luminance(c, a, r):
    return c * np.log(1 + a * r)


# input * transformation matrix (matrix mul)
def colour_change(pixels, transformation_matrix):
    return pixels @ transformation_matrix.T


def dog_vision_array_transform(frame):
    """
    Applies a rough 'deuteranopia-like' transform to an RGB frame (NumPy array).
    
    returns an array of the same shape/dtype
    """
    frame_float = frame.astype(np.float32) / 255.0

    h, w, c = frame_float.shape
    pixels = frame_float.reshape(-1, c)

    transformed = colour_change(pixels, DOG_TRANSFORMATION_MATRIX)
    # transformed = luminance(10, 0.1, transformed)

    transformed = transformed.reshape(h, w, c)  # Reshape back to H x W x C for spatial operations
    for channel in range(3):  # Apply blur channel-wise
        transformed[..., channel] = gaussian_filter(transformed[..., channel], sigma=1)
    
    # # Apply fisheye warp
    # transformed = fisheye_warp(transformed, h, w, strength=1.5)

    # get back to output format
    transformed = transformed.reshape(h, w, c)
    transformed = np.clip(transformed, 0, 1)
    transformed_uint8 = (transformed * 255).astype(np.uint8)

    return transformed_uint8

def rgb_to_yuv(img):
    """
    Convert an RGB image to YUV format using NumPy.
    """
    # Define the transformation matrix
    transform_matrix = np.array([[0.299, 0.587, 0.114],
                                 [-0.14713, -0.28886, 0.436],
                                 [0.615, -0.51499, -0.10001]])
    
    # Reshape the image to (N, 3) where N is the number of pixels
    img_flat = img.reshape(-1, 3)
    
    # Apply the transformation
    yuv_flat = img_flat.dot(transform_matrix.T)
    
    # Reshape back to the original image shape
    yuv_img = yuv_flat.reshape(img.shape)
    
    return yuv_img


def yuv_to_rgb(yuv_img):
    """
    Convert a YUV image to RGB format using NumPy.
    """
    # Define the inverse transformation matrix
    inverse_transform_matrix = np.array([[1.0, 0.0, 1.13983],
                                         [1.0, -0.39465, -0.58060],
                                         [1.0, 2.03211, 0.0]])
    
    # Reshape the image to (N, 3) where N is the number of pixels
    yuv_flat = yuv_img.reshape(-1, 3)
    
    # Apply the inverse transformation
    rgb_flat = yuv_flat.dot(inverse_transform_matrix.T)
    
    # Reshape back to the original image shape
    rgb_img = rgb_flat.reshape(yuv_img.shape)
    
    # Clip the values to be in the valid range [0, 1]
    rgb_img = np.clip(rgb_img, 0, 1)
    
    return rgb_img


def cat_vision_array_transform(frame):
    """
    Applies a rough 'deuteranopia-like' transform to an RGB frame (NumPy array).
    
    returns an array of the same shape/dtype
    """
    frame_float = frame.astype(np.float32) / 255.0

    h, w, c = frame_float.shape
    pixels = frame_float.reshape(-1, c)

    transformed = colour_change(pixels, CAT_TRANSFORMATION_MATRIX)
    yuv_image = rgb_to_yuv(transformed)

    # print(transformed.shape)
    # print(transformed)

    transformed = luminance(80, 0.1, transformed)

    transformed = transformed.reshape(h, w, c)  # Reshape back to H x W x C for spatial operations
    for channel in range(3):  # Apply blur channel-wise
        transformed[..., channel] = gaussian_filter(transformed[..., channel], sigma=1)
    
    # # Apply fisheye warp
    # transformed = fisheye_warp(transformed, h, w, strength=1.5)

    # get back to output format
    transformed = transformed.reshape(h, w, c)
    transformed = np.clip(transformed, 0, 1)
    transformed_uint8 = (transformed * 255).astype(np.uint8)

    return transformed_uint8



def simulate_dog_vision_video(input_video_path, output_video_path):
    """
    Reads a video, applies the dog-vision color transform to each frame,
    and writes out a new video.
    """
    # Load the video
    clip = VideoFileClip(input_video_path)
    
    # Apply frame-by-frame transform using fl_image
    dog_vision_clip = clip.fl_image(dog_vision_array_transform)
    
    # Write the transformed clip to a file
    dog_vision_clip.write_videofile(output_video_path, codec="libx264")


def simulate_cat_vision_video(input_video_path, output_video_path):
    """
    Reads a video, applies the dog-vision color transform to each frame,
    and writes out a new video.
    """
    # Load the video
    clip = VideoFileClip(input_video_path)
    
    # Apply frame-by-frame transform using fl_image
    dog_vision_clip = clip.fl_image(cat_vision_array_transform)
    
    # Write the transformed clip to a file
    dog_vision_clip.write_videofile(output_video_path, codec="libx264")