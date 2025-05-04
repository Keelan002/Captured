import cv2
import numpy as np

# Mouse callback function with context object
def select_corner(event, x, y, flags, param):
    context = param  # param is now a dictionary with our context
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # If we already have 4 corners, reset
        if len(context['corners']) >= 4:
            context['corners'] = []
            context['image_copy'] = context['original_image'].copy()
        
        # Add the new corner
        context['corners'].append((x, y))
        
        # Draw the point
        cv2.circle(context['image_copy'], (x, y), 5, (0, 0, 255), -1)
        
        # Add a label
        cv2.putText(context['image_copy'], f"{len(context['corners'])}", (x+10, y+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # If we have at least 2 points, draw lines between them
        if len(context['corners']) >= 2:
            for i in range(1, len(context['corners'])):
                cv2.line(context['image_copy'], context['corners'][i-1], context['corners'][i], (0, 255, 0), 2)
            
            # If we have 4 points, connect the last to the first
            if len(context['corners']) == 4:
                cv2.line(context['image_copy'], context['corners'][3], context['corners'][0], (0, 255, 0), 2)
        
        # Update the display
        cv2.imshow("Select Corners", context['image_copy'])

def manual_corner_selection(image):
    # Create a context object to store the state
    context = {
        'corners': [],
        'image_copy': image.copy(),
        'original_image': image.copy()
    }
    
    # Create a window and set the mouse callback with context
    cv2.namedWindow("Select Corners")
    cv2.setMouseCallback("Select Corners", select_corner, context)
    
    # Display instructions
    print("Click on the 4 corners of the chessboard playing area (the 8x8 grid).")
    print("Click in this order: top-left, top-right, bottom-right, bottom-left.")
    print("Press 'r' to reset, 'c' to confirm selection, or 'q' to quit.")
    
    # Show the image
    cv2.imshow("Select Corners", context['image_copy'])
    
    # Wait for keyboard input
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        # Reset on 'r'
        if key == ord('r'):
            context['corners'] = []
            context['image_copy'] = context['original_image'].copy()
            cv2.imshow("Select Corners", context['image_copy'])
        
        # Confirm on 'c'
        elif key == ord('c'):
            if len(context['corners']) == 4:
                break
            else:
                print(f"Please select exactly 4 corners. You have {len(context['corners'])} currently.")
        
        # Quit on 'q'
        elif key == ord('q'):
            context['corners'] = []
            break
    
    cv2.destroyAllWindows()
    
    # Convert to appropriate format if needed
    if context['corners']:
        return np.array(context['corners'], dtype=np.float32)
    else:
        return None

# Cached corners - this will store the corners once they're selected
_cached_corners = None

def get_corners(image, use_cached=True):
    global _cached_corners
    
    # If we have cached corners and should use them, return them
    if use_cached and _cached_corners is not None:
        print("Using previously selected corners.")
        return True, _cached_corners
    
    # Otherwise, prompt the user to select corners
    selected_corners = manual_corner_selection(image)
    found = selected_corners is not None
    
    if found:
        """ print("Selected corners:")
        for i, corner in enumerate(selected_corners):
            print(f"Corner {i+1}: {corner}") """
        
        # Cache the corners for future use
        _cached_corners = selected_corners
    else:
        print("Corner selection canceled.")
        selected_corners = None

    return found, selected_corners