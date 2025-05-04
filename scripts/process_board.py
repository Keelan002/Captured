import os
import cv2
import numpy as np


def detect_piece_color(square_image):
    """
    Detect piece color using HSV color space which better separates color from brightness.
    """
    # Convert to BGR if grayscale
    if len(square_image.shape) == 2:
        square_image = cv2.cvtColor(square_image, cv2.COLOR_GRAY2BGR)
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(square_image, cv2.COLOR_BGR2HSV)
    
    # Get center region
    margin = 40
    height, width = hsv.shape[:2]
    if height <= 2*margin or width <= 2*margin:
        center_hsv = hsv
    else:
        center_hsv = hsv[margin:height-margin, margin:width-margin]

    #cv2.imshow("CENTRE", center_hsv)
    #cv2.waitKey(0)
    
    # Extract the V channel (brightness)
    v_channel = center_hsv[:,:,2]
    
    # Use Otsu's method to find optimal threshold
    _, binary = cv2.threshold(v_channel, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # Count white pixels in binary image
    white_percentage = np.sum(binary > 0) / binary.size
    
    # Calculate mean brightness of the V channel
    mean_v = np.mean(v_channel)
    
    # Decision based on both percentage of bright pixels and mean brightness
    #for test6 use 140
    #for FBV use 100
    if  mean_v > 100:
        return 'white', mean_v
    else:
        return 'black', mean_v
    

def extract_squares(image, corners):
    # Make sure corners are in the right format
    corners = np.array(corners, dtype="float32")
    
    # Order corners: top-left, top-right, bottom-right, bottom-left
    # Instead of automatic ordering, let's assume corners are clicked in a specific order
    # If the user clicks in a consistent order, you can use their order directly
    
    # For debugging
    #print("Input corners:", corners)
    
    width = height = 800  # Output size
    square_size = width // 8  # Size of each square
    
    # Define destination points for perspective transform
    dst = np.array([
        [0, 0],                # top-left
        [width - 1, 0],        # top-right
        [width - 1, height - 1],  # bottom-right
        [0, height - 1]        # bottom-left
    ], dtype="float32")
    
    # Calculate perspective transform matrix
    matrix = cv2.getPerspectiveTransform(corners, dst)
    
    # Apply the transform to get a top-down view
    warped = cv2.warpPerspective(image, matrix, (width, height))
    
    # Save the warped board for debugging
    cv2.imwrite("images/warped_board.jpg", warped)
    
    # Extract individual squares
    squares = []
    for row in range(8):
        row_squares = []
        for col in range(8):
            # Calculate square position
            x = col * square_size
            y = row * square_size
            
            # Extract the square from the warped image
            square = warped[y:y + square_size, x:x + square_size]
            
            # Save a few squares for debugging
            if row == 0 and col == 0:
                cv2.imwrite("images/square_0_0.jpg", square)
            if row == 7 and col == 7:
                cv2.imwrite("images/square_7_7.jpg", square)
                
            row_squares.append(square)
        
        squares.append(row_squares)
    
    return squares, warped

def is_square_white(square_name):

    col = ord(square_name[0].lower()) - ord('a')  # Convert 'a'-'h' to 0-7
    row = int(square_name[1]) - 1  # Convert 1-8 to 0-7
    
    # If sum of row and column is even, square is black
    # If sum is odd, square is white
    return (row + col) % 2 == 1

import cv2
import numpy as np

# Calculate pixel distribution metrics
def calculate_pixel_distribution(binary_image):
    # Count white pixels
    white_pixels = np.where(binary_image > 0)
    if len(white_pixels[0]) == 0:
        return 0  # No white pixels
    
    # Calculate center of mass of white pixels
    center_y = np.mean(white_pixels[0])
    center_x = np.mean(white_pixels[1])
    
    # Calculate average distance from center of mass
    distances = np.sqrt((white_pixels[0] - center_y)**2 + (white_pixels[1] - center_x)**2)
    avg_distance = np.mean(distances)
    
    # Calculate standard deviation of distances
    std_distance = np.std(distances)
    
    # Normalize by image dimensions for consistency
    h, w = binary_image.shape
    normalized_avg_distance = avg_distance / (np.sqrt(h**2 + w**2))
    
    # Calculate clustering ratio (lower means more clustered)
    clustering_ratio = normalized_avg_distance * std_distance / (np.sum(binary_image > 0) / (h * w))
    
    return clustering_ratio

def detect_piece_in_square(square_image, threshold, square_name):
    """
    Detect if a chess piece is present in a square using edge detection,
    with improved preprocessing for better handling of white squares.
    
    Parameters:
        square_image: Image of a single chess square
        threshold: Threshold for edge percentage to consider a square occupied
        square_name: Optional name of the square for debugging
        
    Returns:
        is_occupied: Boolean indicating if a chess piece is present
        confidence: Confidence score (0.0-1.0) for the detection
        debug_image: Visualization of the detection process
    """
    import cv2
    import numpy as np
    
    height, width = square_image.shape[:2]
    debug_image = square_image.copy()
    
    # Define margins to avoid edge spillover (10% from each side)
    margin_x = int(width * 0.1)
    margin_y = int(height * 0.1)
    
    # Extract central portion of the square for analysis (with margins on all sides)
    # Only analyze the top 60% of the square (where pieces are most visible)
    top_portion_height = int(height * 0.6)
    roi = square_image[margin_y:top_portion_height, margin_x:width-margin_x]
    
    # Convert to grayscale
    gray_portion = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Determine if it's likely a white square
    is_white_square = is_square_white(square_name)
    
    # Apply stronger blur for white squares to reduce texture noise
    if is_white_square:
        gray_portion = cv2.GaussianBlur(gray_portion, (7, 7), 0)
    else:
        gray_portion = cv2.GaussianBlur(gray_portion, (5, 5), 0)
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3,3))
    gray_enhanced = clahe.apply(gray_portion)
    
    # Apply binarization using adaptive thresholding with parameters adjusted for square color
    if is_white_square:
        # For white squares, use higher C value to reduce noise detection
        binary = cv2.adaptiveThreshold(
            gray_enhanced, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            11,  # Block size
            3    # Higher C value for white squares to reduce noise
        )
    else:
        # For dark squares, use normal settings
        binary = cv2.adaptiveThreshold(
            gray_enhanced, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            11,  # Block size
            2    # Normal C value for dark squares
        )
    
    # Apply more aggressive morphological operations for white squares
    if is_white_square:
        # Use larger kernel for white squares to remove more noise
        kernel = np.ones((4, 4), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    else:
        # Standard kernel for dark squares
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out tiny contours (more aggressively for white squares)
    min_contour_area = 25 if is_white_square else 10
    significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    
    # Calculate total white pixel percentage
    portion_area = binary.shape[0] * binary.shape[1]
    white_pixel_percentage = np.sum(binary > 0) / portion_area if portion_area > 0 else 0
    
    # Find largest connected component (for piece detection)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    largest_component_area = 0
    if num_labels > 1:  # Skip background label (0)
        # Find the largest non-background component
        component_areas = stats[1:, cv2.CC_STAT_AREA]  # Skip background
        largest_component_area = np.max(component_areas) if len(component_areas) > 0 else 0
    
    # Calculate total contour length
    total_contour_length = sum(cv2.arcLength(cnt, True) for cnt in significant_contours)
    normalized_contour_length = total_contour_length / (width * height) * 1000
    
    # Compute weighted confidence score
    component_weight = 0.5       # Largest component is a strong indicator
    contour_weight = 0.15        # Number of contours
    length_weight = 0.2          # Contour length
    pixel_weight = 0.15          # White pixel percentage
    
    # Normalize each metric to 0-1 scale
    component_confidence = min(1.0, largest_component_area / 150)
    contour_confidence = min(1.0, len(significant_contours) / 3)
    length_confidence = min(1.0, normalized_contour_length / 60)
    pixel_confidence = min(1.0, white_pixel_percentage * 20)
    
    # Apply boosting for combinations of metrics
    boost = 0
    if component_confidence > 0.3 and length_confidence > 0.3:
        boost += 0.1
    if component_confidence > 0.3 and contour_confidence > 0.3:
        boost += 0.1
    if length_confidence > 0.3 and pixel_confidence > 0.3:
        boost += 0.1
    
    # Calculate weighted average
    weighted_confidence = (
        component_weight * component_confidence +
        contour_weight * contour_confidence +
        length_weight * length_confidence +
        pixel_weight * pixel_confidence
    ) + boost
    
    # Ensure confidence is between 0 and 1
    confidence = max(0.0, min(1.0, weighted_confidence))
    
    # Determine if a piece is present based on multiple metrics
    # Use different thresholds for white and dark squares
    if is_white_square:
        is_occupied = (largest_component_area > 50 or 
                      normalized_contour_length > 2 or 
                      (len(significant_contours) >= 1 and white_pixel_percentage >= 0.08))
    else:
        is_occupied = (largest_component_area > 60 or 
                      normalized_contour_length > 5 or 
                      (len(significant_contours) >= 1 and white_pixel_percentage >= 0.05))
    
    """ # Force decision based on confidence for consistency
    if confidence > 0.6:
        is_occupied = True
    elif confidence < 0.2:
        is_occupied = False """
    #for FBV use 0.3, FBV2 use 0.5, for FBV4 use 
    if confidence > 0.5: 
        is_occupied = True 
    else: 
        is_occupied = False
    
    # Create the debugging visualizations
    # Convert each stage to BGR for concatenation
    gray_bgr = cv2.cvtColor(gray_portion, cv2.COLOR_GRAY2BGR)
    enhanced_bgr = cv2.cvtColor(gray_enhanced, cv2.COLOR_GRAY2BGR)
    binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    # Draw contours on binary view
    contour_view = binary_bgr.copy()
    cv2.drawContours(contour_view, significant_contours, -1, (0, 255, 255), 1)
    
    # Create a side-by-side panel showing the steps
    target_height = gray_bgr.shape[0]
    target_width = gray_bgr.shape[1]
    debug_pipeline = np.hstack((gray_bgr, enhanced_bgr, contour_view))
    
    cv2.putText(debug_pipeline, "Gray", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
    cv2.putText(debug_pipeline, "CLAHE", (target_width+10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
    cv2.putText(debug_pipeline, "Contours", (2*target_width+10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
    
    # Add text showing square type
    square_type = "White" if is_white_square else "Dark"
    
    
    # Calculate available width inside debug_image
    available_width = width - 2 * margin_x
    
    # Resize debug_pipeline to match available width
    debug_pipeline_resized = cv2.resize(debug_pipeline, (available_width, debug_pipeline.shape[0]))
    
    # Insert pipeline into debug image
    debug_image[margin_y:margin_y+debug_pipeline.shape[0], margin_x:margin_x+available_width] = debug_pipeline_resized
    
    # Add metrics to debug image
    cv2.putText(debug_image, f"Area: {largest_component_area:.0f}", (5, height-50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(debug_image, f"Len: {normalized_contour_length:.1f}", (5, height-40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(debug_image, f"Cnt: {len(significant_contours)}", (5, height-30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(debug_image, f"Conf: {confidence:.2f}", (5, height-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(debug_image, f"Edg: {white_pixel_percentage:.2f}", (5, height-15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(debug_image, f"Col: {square_type}", (10, height-20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    
    # Draw square name if provided
    if square_name:
        cv2.putText(debug_image, square_name, (5, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return is_occupied, confidence, debug_image


def is_square_occupied(square_image, square_name):
    is_white_square = is_square_white(square_name)
    
    # Use a very low threshold to detect any meaningful contours
    threshold = 0.15  # Extremely sensitive - almost any contour will trigger detection
    
    # Call detect_piece_in_square with the appropriate threshold
    is_occupied, confidence, debug_image = detect_piece_in_square(square_image, threshold, square_name)
    
    # Add additional information to the debug image
    status = "OCCUPIED" if is_occupied else "EMPTY"
    color = (0, 0, 255) if is_occupied else (0, 255, 0)
    square_color = "WHITE" if is_white_square else "BLACK"

    cv2.putText(debug_image, status, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.putText(debug_image, square_color, (5, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return is_occupied, debug_image


def analyze_all_squares(squares_list, output_folder="occupied_squares"):
    """
    Analyze all squares from the provided list of square images, detect piece colors,
    and save occupied squares to a folder.
    
    Args:
        squares_list: 8x8 list of square images
        output_folder: Folder where occupied square images will be saved
    
    Returns:
        occupied_squares_with_color: List of tuples (square_name, color) for occupied squares
        debug_grid: Combined visualization of all squares with analysis
        all_debug_images: Dictionary of debug images for each square
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")
    
    occupied_squares_with_color = []  # Will store (square_name, color) pairs
    all_debug_images = {}

    # Create a composite debug image
    debug_grid = np.zeros((800, 800, 3), dtype=np.uint8)
    square_size = 100  # 800 // 8

    # Iterate through each row and column
    for row in range(8):
        for col in range(8):
            # Generate square name in chess notation
            square_name = f"{chr(ord('a') + col)}{8 - row}"
            
            # Get square image from the list
            square_img = squares_list[row][col]

            #print(f"Analyzing {square_name}")
            # Analyze the square
            is_occupied, debug_img = is_square_occupied(square_img, square_name)

            # Store debug image
            all_debug_images[square_name] = debug_img

            # If occupied, detect color and add to list
            if is_occupied:
                # Detect the piece color
                piece_color, avg_brightness = detect_piece_color(square_img)
                #print(f"square name {square_name} and brightness {avg_brightness}")
                #cv2.imshow("square img", square_img)
                #cv2.waitKey(0)
                
                # Add square and color to the results
                occupied_squares_with_color.append((square_name, piece_color))
                
                # Create a color-annotated debug image
                color_debug_img = debug_img.copy()
                color_text = f"Color: {piece_color, avg_brightness}"
                cv2.putText(color_debug_img, color_text, 
                            (5, -5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
                            (0, 0, 255) if piece_color == 'black' else (255, 0, 0), 1)
                
                # Save images to the output folder
                original_path = os.path.join(output_folder, f"{square_name}_original.png")
                debug_path = os.path.join(output_folder, f"{square_name}_debug.png")
                color_debug_path = os.path.join(output_folder, f"{square_name}_color_debug.png")
                
                #cv2.imwrite(original_path, square_img)
                #cv2.imwrite(debug_path, debug_img)
                #cv2.imwrite(color_debug_path, color_debug_img)
                
                #print(f"Saved occupied square {square_name} (color: {piece_color}) to {original_path}")
                
                
                # Show the square with color annotation (optional)
                display_img = cv2.resize(color_debug_img, (200, 200))
                """ cv2.imshow(f"Occupied Square: {square_name} ({piece_color})", display_img)
                cv2.waitKey(1) """

            # Add to debug grid
            y = row * square_size
            x = col * square_size
            
            # If occupied, add color label to the debug grid
            if is_occupied:
                grid_piece = cv2.resize(debug_img, (square_size, square_size))
                color = occupied_squares_with_color[-1][1]  # Get the last added color
                cv2.putText(grid_piece, color[0].upper(), 
                           (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                           (0, 0, 255) if color == 'black' else (255, 0, 0), 1)
                debug_grid[y:y + square_size, x:x + square_size] = grid_piece
            else:
                debug_grid[y:y + square_size, x:x + square_size] = cv2.resize(debug_img, (square_size, square_size))
    
    # Save the full debug grid
    grid_path = os.path.join(output_folder, f"full_board_debug.png")
    cv2.imwrite(grid_path, debug_grid)
    print(f"Saved full board debug visualization to {grid_path}")
    
    """ # Print summary of findings
    print("\nSummary of occupied squares:")
    for square, color in occupied_squares_with_color:
        print(f"  {square}: {color} piece") """
    
    # Display the debug grid (optional)
    #cv2.imshow("Board Analysis", cv2.resize(debug_grid, (600, 600)))
    #cv2.waitKey(1)
    
    return occupied_squares_with_color, debug_grid, all_debug_images
