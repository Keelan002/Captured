import scripts.create_PGN
import scripts.infer_move
import scripts.process_board as process_board
import scripts.infer_move as infer_move
import scripts.board_state_class as bs
import cv2
import scripts.process_video as pv
import scripts.create_PGN as pgn
import scripts.get_board_corners as get_board_corners
import scripts.create_board_state
import os
import time



def resize_image(image):
    # Resize if image is too large
    max_dimension = 800
    height, width = image.shape[:2]
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height))
        #print(f"Resized image to: {image.shape}")

    return image


import time
import os
import cv2
import numpy as np



def process_images(frames):
    os.makedirs("debug_frames", exist_ok=True)
    board_states = []
    moves = []
    previous_board_state = None  # Initialize the previous board state
    previous_occupied_squares = None  # Store previous occupied squares
    
    # Lists for timing statistics
    processing_times = []
    
    # Process each image
    for i, image in enumerate(frames):
        # Start timing for this frame
        frame_start_time = time.time()  # Start overall frame timer
        
        print(f"\nProcessing image {i+1} of {len(frames)}")
        
        # Track individual steps
        step_times = {}
        
        # Time the resizing step
        resize_start = time.time()
        resized_image = resize_image(image)
        step_times['resize'] = time.time() - resize_start
        
        # Time the corner detection step
        corner_start = time.time()
        use_cached = (i > 0)
        found, corners = get_board_corners.get_corners(resized_image, use_cached=use_cached)
        step_times['corner_detection'] = time.time() - corner_start

        if found:
            print("Chessboard detected!")

            # Time the square extraction
            extraction_start = time.time()
            squares, warped = process_board.extract_squares(resized_image, corners)
            step_times['square_extraction'] = time.time() - extraction_start
            print("Squares have been extracted")

            # Time the square analysis
            analysis_start = time.time()
            occupied_squares, debug_grid, all_debug_images = process_board.analyze_all_squares(squares)
            step_times['square_analysis'] = time.time() - analysis_start

            #cv2.imshow(f"Analysis Grid - Image {i+1}", debug_grid)
          
            # Time the validation step
            validation_start = time.time()
            if not pv.is_valid_frame(occupied_squares, previous_occupied_squares):
                print(f"Skipping frame {i+1} (invalid)")
                
                # Calculate total time even for skipped frames
                frame_total_time = time.time() - frame_start_time
                processing_times.append((i, frame_total_time, step_times, "skipped-invalid"))
                
                continue  # Skip to next frame
            else:
                print("valid frame")
            step_times['validation'] = time.time() - validation_start
            
            
            
            # Time the board state processing
            board_state_start = time.time()
            if i == 0:  # Is first frame
                print('FIRST FRAME - INITIAL BOARD STATE')
                board_state = scripts.create_board_state.construct_initial_board_state()
                step_times['board_state'] = time.time() - board_state_start
                status = "initial"
            else:
                # For subsequent frames, we detect the move
                print('DETECTING MOVE FROM PREVIOUS FRAME')
                # Use the previously stored board state with the updated function
                current_board_state, source_square, destination_square, is_capture, captured_piece, is_castle, castle_type, is_not_valid = scripts.create_board_state.create_current_board_state(
                    occupied_squares, previous_board_state)
                
                if is_not_valid == True:
                    print("Too many colour chanages, skipping")
                    continue

                # Store current occupied squares for next iteration
                previous_occupied_squares = occupied_squares
                board_state = current_board_state

                #print(f"BOARD STATE{board_state}")
                
                # Only process the move if one was detected
                if source_square and destination_square:
                    move_inference_start = time.time()
                    print(f"Source square: {source_square}")
                    print(f"Destination square: {destination_square}")
                    print(f"Is capture: {is_capture}")
                    print(f"Is castle: {is_castle}")
                    
                    if is_capture:
                        print(f"Captured piece: {captured_piece['piece']} ({captured_piece['color']})")
                    
                    # Record the move
                    move = scripts.infer_move.record_move(
                        source_square, 
                        destination_square,
                        previous_board_state,
                        is_capture,
                        captured_piece,
                        is_castle,
                        castle_type
                    )
                    moves.append(move)
                    step_times['move_inference'] = time.time() - move_inference_start
                    step_times['board_state'] = time.time() - board_state_start
                    status = "move-detected"
                else:
                    print("No move detected in this frame")
                    step_times['board_state'] = time.time() - board_state_start
                    
                    # Calculate total time for frames with no move
                    frame_total_time = time.time() - frame_start_time
                    processing_times.append((i, frame_total_time, step_times, "no-move"))
                    
                    continue  # Skip this frame for board state updating
            
            # Store this board state for the next iteration
            previous_board_state = board_state
        
        else:
            print(f"No chessboard detected in image {i+1}")
            status = "no-chessboard"
        
        # Calculate total processing time for this frame
        frame_total_time = time.time() - frame_start_time
        processing_times.append((i, frame_total_time, step_times, status))
        
        # Print timing information for this frame
        print(f"\nFrame {i+1} processing time: {frame_total_time:.4f} seconds")
        for step, duration in step_times.items():
            print(f"  - {step}: {duration:.4f}s ({duration/frame_total_time*100:.1f}%)")
    
    # Print overall timing statistics
    if processing_times:
        print("\n--- Processing Time Statistics ---")
        total_frames = len(processing_times)
        total_time = sum(time[1] for time in processing_times)
        avg_time = total_time / total_frames
        
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Average time per frame: {avg_time:.4f} seconds")
        print(f"Fastest frame: {min(time[1] for time in processing_times):.4f} seconds")
        print(f"Slowest frame: {max(time[1] for time in processing_times):.4f} seconds")
        
        # Calculate average time by status
        status_times = {}
        for _, time_taken, _, status in processing_times:
            if status not in status_times:
                status_times[status] = []
            status_times[status].append(time_taken)
        
        print("\nAverage times by frame status:")
        for status, times in status_times.items():
            print(f"  - {status}: {sum(times)/len(times):.4f} seconds (from {len(times)} frames)")
        
        # Calculate average time for each processing step
        step_averages = {}
        step_count = {}
        
        for _, _, steps, _ in processing_times:
            for step, duration in steps.items():
                if step not in step_averages:
                    step_averages[step] = 0
                    step_count[step] = 0
                step_averages[step] += duration
                step_count[step] += 1
        
        print("\nAverage times by processing step:")
        for step in step_averages:
            avg = step_averages[step] / step_count[step]
            print(f"  - {step}: {avg:.4f} seconds ({avg/avg_time*100:.1f}% of frame time)")
    
    return board_states, moves

def post_process_moves(moves):
    """
    Post-process the detected moves to identify and combine castling moves.
    
    Args:
        moves: List of move dictionaries
        
    Returns:
        Processed list with castling properly identified
    """
    processed_moves = []
    i = 0
    
    while i < len(moves):
        current_move = moves[i]
        
        # Check if this could be the first part of a castling move (king moving two squares)
        if (current_move['piece'].upper() == 'K' and 
            i + 1 < len(moves) and 
            moves[i+1]['piece'].upper() == 'R' and
            current_move['color'] == moves[i+1]['color']):
            
            king_move = current_move
            rook_move = moves[i+1]
            
            # Check for kingside castling
            if ((king_move['from'] == 'e1' and king_move['to'] == 'g1' and 
                 rook_move['from'] == 'h1' and rook_move['to'] == 'f1') or
                (king_move['from'] == 'e8' and king_move['to'] == 'g8' and 
                 rook_move['from'] == 'h8' and rook_move['to'] == 'f8')):
                
                # Create a castling move
                castle_move = {
                    'from': king_move['from'],
                    'to': king_move['to'],
                    'piece': king_move['piece'],
                    'color': king_move['color'],
                    'is_castle': True,
                    'castle_type': 'short'
                }
                processed_moves.append(castle_move)
                i += 2  # Skip both moves
                continue
                
            # Check for queenside castling
            elif ((king_move['from'] == 'e1' and king_move['to'] == 'c1' and 
                   rook_move['from'] == 'a1' and rook_move['to'] == 'd1') or
                  (king_move['from'] == 'e8' and king_move['to'] == 'c8' and 
                   rook_move['from'] == 'a8' and rook_move['to'] == 'd8')):
                
                # Create a castling move
                castle_move = {
                    'from': king_move['from'],
                    'to': king_move['to'],
                    'piece': king_move['piece'],
                    'color': king_move['color'],
                    'is_castle': True,
                    'castle_type': 'long'
                }
                processed_moves.append(castle_move)
                i += 2  # Skip both moves
                continue
        
        # If not castling, add the move as is
        processed_moves.append(current_move)
        i += 1
    
    return processed_moves

def run():
    
    # Or extract a specific number of frames
    num_frames, frames = pv.extract_frames(
        video_path="videos/FBV8.mp4",
        output_dir="extracted_frames",
        frames_interval= 1
    )
    print('num frames extracted', num_frames)

    board_states, moves = process_images(frames)

    post_processed_moves = post_process_moves(moves)

    """ for board in board_states:
        print(board) """

    pgn = scripts.create_PGN.create_pgn(post_processed_moves)

    return pgn




def main():
    pgn = run()
    
    

if __name__ == "__main__":
    main()



# use setup to tracck where pieces are currently 