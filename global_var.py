import os

"""
Global variables and configuration for the depth mapping project.

This module defines project-wide constants and paths used across
the training and evaluation pipeline.
"""

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(ROOT_PATH, 'log')