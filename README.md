# apple-detection-cv2
Detecting and creating bounding boxes for apples in an image without neural networks.

The first step in the detection algorithm is to detect circular objects in the image using Circle Hough Transform.

The image is converted to grayscale, and a Gaussian Blur with a 15x15 sized Kernel is applied to the original image. The kernel size was decided on through manual trial and error, seeking to find a kernel size that maximized the number of apples correctly identified while minimizing the number of false positives when using the Circle Hough Transform. The Circle Hough Transform itself takes the blurred image as input, and uses the hyperparameters
dp=1 


“Inverse ratio of the accumulator resolution to the image resolution. For example, if dp=1 , the accumulator has the same resolution as the input image. If dp=2 , the accumulator has half as big width and height. For HOUGH_GRADIENT_ALT the recommended value is dp=1.5, unless some small very circles need to be detected.” Source: opencv documentation
 
dp=1 was chosen as the size of the apples are unknown

minDist=75

The minimum distance (in pixels) between the centers of the detected circles

param1=50

Method-specific parameter for edge detection (higher = stricter)

param2=40

Second method-specific parameter for circle center detection (lower = stricter)

minRadius= 30

The minimum radius (in pixels) of the circles.

maxRadius= 800

The maximum radius (in pixels) of the circles.

After the circles are identified using the Circle Hough Transform, a function is called to perform further validation to determine whether the circles represent an apple.

The first way the identified circles are filtered is by checking to see whether a circle is fully contained inside another circle. In the event of a circle being fully contained inside another circle, the inside circle is not needed as the outer one fully encapsulates it. 
This is done by comparing the centers between a circle with every other circle, and seeing if the distance between the centers is smaller than the difference in radius between two circles. If the distance between two circles 1 and 2 is lower than the absolute difference in radii between the two circles, it means the circle we are comparing is entirely contained within another circle.

The second way the circles are filtered is by colour, where the Red, Green and Blue channels are extracted from the image in each detection zone and compared with a certain threshold of pixels to be considered an apple. This is done by first using a mask image filled with black pixels to map the circles onto as white pixels, and the final circles are extracted by using the bitwise AND operator between the original image and the mask. This is to isolate the pixels in the original image that are contained within a detection circle, while setting all other areas to black. These pixels are then compared with the colour thresholds to ensure there are minimum amounts of blue pixels, while having red or green pixels.
