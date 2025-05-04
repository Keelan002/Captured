import cv2
import numpy as np
from typing import Optional, List, Tuple
from PIL import Image, ExifTags
import cv2
import os
from typing import Optional
import time


def needs_rotation(frame):
        
    im1 = Image.open(frame)
    
    print(f'im1.size={im1.size}')
    
    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation] == 'Orientation':
            break

    exif1 = dict(im1.getexif().items())
    
    print(f'exif1 orientation={exif1[orientation]}')
    return exif1[orientation == 1]
    

def rotate_frames(frame):
    angle = -90
    r_img = frame.rotate(angle)
    print("Frame rotated")
    return r_img
    


import time
import os
import cv2
import numpy as np
from typing import Optional, Tuple

def extract_frames(
    video_path: str,
    output_dir: str,
    frames_interval: Optional[float] = None,
    total_frames: Optional[int] = None
) -> Tuple[int, list]:
    """
    Extract frames from a video file based on either a time interval or desired total number of frames.
    
    Args:
        video_path: Path to the input video file
        output_dir: Directory where extracted frames will be saved
        frames_interval: Time interval between frames in seconds (mutually exclusive with total_frames)
        total_frames: Desired total number of frames to extract (mutually exclusive with frames_interval)
    
    Returns:
        tuple containing:
            - number of frames extracted
            - list of extracted frames as numpy arrays
    """
    # Start overall timing
    overall_start_time = time.time()
    timing_data = {}
    
    # Input validation
    validation_start = time.time()
    if (frames_interval is None and total_frames is None) or \
       (frames_interval is not None and total_frames is not None):
        raise ValueError("Exactly one of frames_interval or total_frames must be provided")
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    timing_data['validation'] = time.time() - validation_start
    
    # Open the video file
    open_start = time.time()
    print(f"Opening video file: {video_path}")
    video = cv2.VideoCapture(video_path)
    timing_data['open_video'] = time.time() - open_start
    print(f"Video opened in {timing_data['open_video']:.4f} seconds")
    
    if not video.isOpened():
        raise RuntimeError(f"Error opening video file: {video_path}")
    
    try:
        # Get video properties
        properties_start = time.time()
        fps = video.get(cv2.CAP_PROP_FPS)
        total_video_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_video_frames / fps
        
        print(f"Video properties: {total_video_frames} frames, {fps:.2f} FPS, {duration:.2f} seconds")
        timing_data['get_properties'] = time.time() - properties_start
        
        # Calculate frame indices
        indices_start = time.time()
        if frames_interval is not None:
            # Calculate frame indices based on time interval
            frame_indices = [
                int(i * fps)
                for i in np.arange(0, duration, frames_interval)
            ]
            print(f"Using time interval: {frames_interval} seconds")
            print(f"Will extract {len(frame_indices)} frames")
        else:
            # Calculate frame indices based on desired total frames
            step = total_video_frames / total_frames
            frame_indices = [
                int(i * step)
                for i in range(total_frames)
            ]
            print(f"Using total frames: {total_frames}")
            print(f"Frame step size: {step:.2f}")
        
        timing_data['calculate_indices'] = time.time() - indices_start
        
        # List to store the extracted frames
        frames = []
        
        # Process frames
        extraction_start = time.time()
        frame_times = []
        
        print(f"Starting frame extraction of {len(frame_indices)} frames...")
        for i, frame_idx in enumerate(frame_indices):
            frame_start = time.time()
            
            # Set video position to desired frame
            seek_start = time.time()
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            seek_time = time.time() - seek_start
            
            # Read the frame
            read_start = time.time()
            ret, frame = video.read()
            read_time = time.time() - read_start
            
            if not ret:
                print(f"Warning: Could not read frame {frame_idx}")
                continue
            
            # Save frame to file
            save_start = time.time()
            frame_filename = os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            save_time = time.time() - save_start
            
            # Add to frames list
            frames.append(frame)
            
            # Calculate frame processing time
            frame_total_time = time.time() - frame_start
            frame_times.append(frame_total_time)
            
            # Print progress every 10 frames or for the first and last frame
            if i == 0 or i == len(frame_indices)-1 or (i+1) % 10 == 0:
                print(f"Frame {i+1}/{len(frame_indices)}: index={frame_idx}, " 
                      f"time={frame_total_time:.4f}s (seek: {seek_time:.4f}s, "
                      f"read: {read_time:.4f}s, save: {save_time:.4f}s)")
        
        timing_data['frame_extraction'] = time.time() - extraction_start
        
        # Calculate overall time
        overall_time = time.time() - overall_start_time
        
        # Print timing summary
        print("\n--- Timing Summary ---")
        print(f"Total frames extracted: {len(frames)}")
        print(f"Overall extraction time: {overall_time:.4f} seconds")
        print(f"Average time per frame: {overall_time/len(frames) if frames else 0:.4f} seconds")
        
        if frame_times:
            print(f"Fastest frame: {min(frame_times):.4f} seconds")
            print(f"Slowest frame: {max(frame_times):.4f} seconds")
        
        print("\nBreakdown by stage:")
        print(f"- Validation and setup: {timing_data['validation']:.4f} seconds")
        print(f"- Opening video: {timing_data['open_video']:.4f} seconds")
        print(f"- Getting video properties: {timing_data['get_properties']:.4f} seconds")
        print(f"- Calculating frame indices: {timing_data['calculate_indices']:.4f} seconds")
        print(f"- Frame extraction: {timing_data['frame_extraction']:.4f} seconds "
              f"({timing_data['frame_extraction']/overall_time*100:.1f}% of total)")
        
        print(f"\nEstimated time for a 10-minute video at 2 frames/second: "
              f"{(overall_time/len(frames))*(10*60*2) if frames else 0:.2f} seconds")
        
        # Return the number of frames and the frames themselves
        return len(frames), frames
    
    finally:
        # Always release the video capture object
        video.release()

def is_valid_frame(current_occupied_squares, previous_occupied_squares, threshold=3):
    """
    Determine if a frame should be considered valid for analysis.
    
    Args:
        current_occupied_squares: List of squares currently detected as occupied
        previous_occupied_squares: List of squares previously detected as occupied
        threshold: Maximum allowed change in number of occupied squares
        
    Returns:
        boolean: True if frame is valid, False if it should be skipped
    """
    # If this is the first frame, it's always valid
    if previous_occupied_squares is None:
        return True
    
    # Check if number of occupied squares changed dramatically
    current_count = len(current_occupied_squares)
    previous_count = len(previous_occupied_squares)
    
    if abs(current_count - previous_count) > threshold:
        print(f"Invalid frame: Occupied square count changed by {abs(current_count - previous_count)}")
        print(f"Current count: {current_count}, Previous count: {previous_count}")
        return False
        
    # Check if there's an unusual number of new squares
    current_squares = {square for square, _ in current_occupied_squares}
    previous_squares = {square for square, _ in previous_occupied_squares}
    
    newly_occupied = current_squares - previous_squares
    newly_empty = previous_squares - current_squares
    
    # Chess moves typically involve one piece moving (one newly empty, one newly occupied)
    # or a capture (one newly empty, potentially none newly occupied)
    # Two newly occupied squares would be invalid
    
    
    if (len(newly_empty), len(newly_occupied)) not in [(0, 0), (1, 1), (1, 0), (2,2)]:
        print(f"Invalid frame: {len(newly_empty)} newly empty and {len(newly_occupied)} newly occupied")
        print(f"Newly empty squares: {sorted(newly_empty)}")
        print(f"Newly occupied squares: {sorted(newly_occupied)}")
        return False
    
    return True


    """ if len(newly_occupied) > 1:
        print(f"Invalid frame: {len(newly_occupied)} newly occupied squares detected")
        print(f"Newly occupied squares: {sorted(newly_occupied)}")
        return False
    
    # Similarly, too many newly empty squares is suspicious
    if len(newly_empty) > 2:  # Allow up to 2 for castling
        print(f"Invalid frame: {len(newly_empty)} newly empty squares detected")
        print(f"Newly empty squares: {sorted(newly_empty)}")
        return False
    
    if len(newly_empty) >= 1 and len(newly_occupied) == 0:
        print(f"Invalid frame: {len(newly_empty)} newly empty and {len(newly_occupied)} newly occuped")
        print(f"Newly empty squares: {sorted(newly_empty)}")
        print(f"Newly occupied squares: {sorted(newly_occupied)}")
        return False
    
    if len(newly_empty) == 0 and len(newly_occupied) > 0:
        print(f"Invalid frame: {len(newly_empty)} newly empty and {len(newly_occupied)} newly occuped")
        print(f"Newly empty squares: {sorted(newly_empty)}")
        print(f"Newly occupied squares: {sorted(newly_occupied)}")
        return False
    if len(newly_occupied) == 0 and len(newly_empty) > 0:
        print(f"Invalid frame: {len(newly_empty)} newly empty and {len(newly_occupied)} newly occuped")
        print(f"Newly empty squares: {sorted(newly_empty)}")
        print(f"Newly occupied squares: {sorted(newly_occupied)}")
        return False """
    