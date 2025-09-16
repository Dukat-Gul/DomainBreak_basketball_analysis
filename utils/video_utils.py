"""
A module for reading and writing video files.

This module provides utility functions to load video frames into memory and save
processed frames back to video files, with support for common video formats.
"""

import cv2
import os

def read_video(video_path):
    """
    Read all frames from a video file into memory.

    Args:
        video_path (str): Path to the input video file.

    Returns:
        list: List of video frames as numpy arrays.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(ouput_video_frames,output_video_path):
    """
    Save a sequence of frames as a video file.

    Creates necessary directories if they don't exist and writes frames using XVID codec.

    Args:
        ouput_video_frames (list): List of frames to save.
        output_video_path (str): Path where the video should be saved.
    """
    # If folder doesn't exist, create it
    if not os.path.exists(os.path.dirname(output_video_path)):
        os.makedirs(os.path.dirname(output_video_path))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(
        output_video_path,
        fourcc,
        24,
        (ouput_video_frames[0].shape[1], ouput_video_frames[0].shape[0]),
    )
    for frame in ouput_video_frames:
        out.write(frame)
    out.release()


def read_video_with_meta(video_path):
    """
    Legge tutti i frame e restituisce anche fps e size utili per il salvataggio.

    Returns: (frames, fps, (width, height))
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps, (width, height)


def save_video_with_fps(frames, output_video_path, fps: float):
    """Salva i frame usando l'fps specificato."""
    if not os.path.exists(os.path.dirname(output_video_path)):
        os.makedirs(os.path.dirname(output_video_path))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(output_video_path, fourcc, float(fps or 24), (w, h))
    for f in frames:
        out.write(f)
    out.release()
