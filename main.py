"""Entry point for the human body tracking project.

For now, verify the base computer-vision libraries can be imported.
"""

import cv2
import mediapipe as mp
import numpy as np

print("OpenCV version:", cv2.__version__)
print("MediaPipe version:", mp.__version__)
print("NumPy version:", np.__version__)
print("Environment setup is complete. You can now use OpenCV, MediaPipe, and NumPy in your project.")
