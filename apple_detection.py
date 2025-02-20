# Functions to detect bounding boxes of apples in an image without using a pre-trained model
import cv2
from cv2 import imshow as imshow
import numpy as np

def detect_apples(image, min_radius=30, max_radius=800, param1=50, param2=40):
    """
    Function to detect apples in an image.
    Params:
    Same as detect_circles()

    Returns:
    # np array, the image with the circles drawn on it,
    # np array, the bounding boxes/circles of the detected apples
    """
    circles = detect_circles(image, min_radius=min_radius, max_radius=max_radius, param1=param1, param2=param2)
    filtered_circles = remove_contained_circles(circles, image)
    return draw_circles(image, filtered_circles), filtered_circles


def detect_circles(image, min_radius=30, max_radius=800, param1=50, param2=40):
    """
    Function to detect circles in an image using the Hough Circle Transform.
    
    Params:
    - image (np array): Input image on which circles need to be detected.
    - min_radius: Minimum radius of the circles to detect.
    - max_radius: Maximum radius of the circles to detect.
    - param1: First method-specific parameter for edge detection (higher is stricter).
    - param2: Second method-specific parameter for circle center detection (lower is stricter).
    
    Returns:
    - np array: Detected circles, each represented by [x, y, r] (center and radius).
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise and improve circle detection
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    
    # Apply the Hough Circle Transform to detect circles
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, dp=1.2, minDist=75,
        param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius
    )
    
    # If circles are detected, round them and return
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        return circles
    return []

def draw_circles(image, circles):
    """
    Function to draw circles on an image.
    
    Params:
    - image (np array): Input image to draw the circles on.
    - circles (np array): Detected circles (each circle is [x, y, r]).
    
    Returns:
    - image_with_circles (np array): Image with circles drawn.
    """
    image_with_circles = image.copy()
    
    for (x, y, r) in circles:
        # Draw the circle outline
        cv2.circle(image_with_circles, (x, y), r, (0, 255, 0), 4)
        # Draw the center of the circle
        cv2.circle(image_with_circles, (x, y), 3, (0, 0, 255), 3)
    
    return image_with_circles

def is_apple_colour(pixel_values, colour_space):
    """
    Params:
    pixel_values (np array): Array of pixel values from the circle area.
    colour_space (str): Colour space to use for the analysis ('RGB' or 'HSV').

    Returns:
    bool: True if the colour distribution suggests the circle is likely an apple.
    """
    if colour_space.upper() == 'RGB':
        return is_apple_colour_rgb(pixel_values)
    elif colour_space.upper() == 'HSV':
        return is_apple_colour_hsv(pixel_values)
    else:
        raise ValueError("Invalid colour space. Choose 'RGB' or 'HSV'.")

def is_apple_colour_hsv(pixel_values, 
                       red_hue_low=10, 
                       red_hue_high=170, 
                       green_hue_lower=35, 
                       green_hue_upper=85, 
                       saturation_threshold=100, 
                       value_threshold=100, 
                       apple_fraction_threshold=0.3,
                       valid_pixel_fraction_threshold=0.5,
                       non_apple_fraction_threshold=0.4):
    """
    Determine whether the color distribution in an HSV region is likely that of an apple,
    supporting both red and green apples while excluding non-apple colors like blue, orange, or black.
    
    Params:
      pixel_values (np array): Array of shape (n, 3) containing HSV pixels (H in [0, 179], S and V in [0, 255])
      red_hue_low: Upper bound for the lower red hue range
      red_hue_high: Lower bound for the upper red hue range
      green_hue_lower: Lower bound for the green apple hue range
      green_hue_upper: Upper bound for the green apple hue range
      saturation_threshold: Minimum saturation for a pixel to be considered valid
      value_threshold: Minimum brightness for a pixel to be considered valid
      apple_fraction_threshold (float): Minimum fraction of valid pixels that must be apple-like (%)
      valid_pixel_fraction_threshold (float): Minimum fraction of pixels that must be valid (%) 
      non_apple_fraction_threshold (float): Maximum fraction of valid pixels allowed in non-apple hues (%)
    
    Returns:
      bool: True if the region is likely an apple, False otherwise.
    """
    # Total number of pixels in the circle
    total_pixels = pixel_values.shape[0]
    
    # Extract HSV channels
    hue = pixel_values[:, 0]
    sat = pixel_values[:, 1]
    val = pixel_values[:, 2]
    
    # Step 1: Filter pixels that are bright and saturated
    valid = (sat >= saturation_threshold) & (val >= value_threshold)
    num_valid = np.sum(valid)
    
    # Check if enough pixels are valid (avoids misclassifying dark objects)
    if num_valid / total_pixels < valid_pixel_fraction_threshold:
        return False
    
    # Step 2: Check for non-apple colors to reduce false positives
    # Orange hues: 10–25
    orange_pixels = valid & (hue >= 10) & (hue <= 25)
    # Blue hues: 100–140
    blue_pixels = valid & (hue >= 100) & (hue <= 140)
    
    orange_fraction = np.sum(orange_pixels) / num_valid
    blue_fraction = np.sum(blue_pixels) / num_valid
    
    # Reject if too many pixels are orange or blue
    if orange_fraction > non_apple_fraction_threshold or blue_fraction > non_apple_fraction_threshold:
        return False
    
    # Step 3: Identify apple-colored pixels
    red_pixels = valid & ((hue < red_hue_low) | (hue > red_hue_high))
    green_pixels = valid & (hue >= green_hue_lower) & (hue <= green_hue_upper)
    
    # Calculate fractions of apple colors among valid pixels
    red_fraction = np.sum(red_pixels) / num_valid
    green_fraction = np.sum(green_pixels) / num_valid
    
    # Classify as apple if sufficient red or green pixels are present
    if red_fraction >= apple_fraction_threshold or green_fraction >= apple_fraction_threshold:
        return True
    
    return False

def is_apple_colour_rgb(pixel_values, red_threshold=100, green_threshold=50, blue_threshold=50):
    """
    Function to check if the color distribution of the pixels suggests it's an apple.
    
    Params:
    - pixel_values (np array): Array of pixel values from the circle area.
    - red_threshold: Minimum red intensity for apple
    - green_threshold: Minimum green intensity for apple
    - blue_threshold: Maximum blue intensity to avoid blue objects
    
    Returns:
    - bool: True if the color distribution suggests the circle is likely an apple.
    """
    # Extract red, green, and blue channels
    red_pixels = pixel_values[:, 2]
    green_pixels = pixel_values[:, 1]
    blue_pixels = pixel_values[:, 0]
    
    # Count pixels that are above the threshold for red and green
    red_pixels_above_threshold = np.sum(red_pixels >= red_threshold)
    green_pixels_above_threshold = np.sum(green_pixels >= green_threshold)
    
    # Count pixels that are below the threshold for blue
    blue_pixels_below_threshold = np.sum(blue_pixels < blue_threshold)
    
    # To be considered an apple, we need enough red and green pixels,
    # and not too many blue pixels
    if red_pixels_above_threshold > 0 and green_pixels_above_threshold > 0:
        # Ensure that blue pixels are not dominant
        if blue_pixels_below_threshold > len(pixel_values) * 0.1:  # Allow up to 20% blue pixels
            return True

    return False

def remove_contained_circles(circles, image):
    """
    Function to remove circles that are fully contained within another circle, 
    and also validate that the circle corresponds to an apple based on its color.
    
    Params:
    - circles (np array): Detected circles (each circle is [x, y, r]).
    - image (np array): The image from which the circles were detected.
    
    Returns:
    - filtered_circles (np array): Circles that are not fully contained within another circle 
      and are likely apples.
    """
    # List to store circles that are not contained and likely apples
    filtered_circles = []
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Iterate over each circle
    for i, (x1, y1, r1) in enumerate(circles):
        is_contained = False
        
        # Create a mask for the current circle
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, (x1, y1), r1, 255, -1)  # Create a circular mask

        # Extract the region inside the circle using the mask
        circle_region = cv2.bitwise_and(image, image, mask=mask)

        # Only keep non-zero pixels (the pixels inside the circle)
        circle_pixels = circle_region[mask != 0]

        # Check if the color distribution suggests it's an apple
        if not is_apple_colour(circle_pixels, "RGB"): # Use RGB
            continue
        
        # Check for containment
        for j, (x2, y2, r2) in enumerate(circles):
            if i == j:
                continue
            
            distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if distance < abs(r1 - r2):
                is_contained = True
                break

        # If the circle is not contained and is a valid apple, add it to the filtered list
        if not is_contained:
            filtered_circles.append((x1, y1, r1))
    
    return np.array(filtered_circles)