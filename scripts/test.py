import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import modules from scripts
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
import cv2
import re

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


pgn_fbv8 = "1.d4 d4 2.c4 e6 3.Nc3 Nfc 4.Bg5 h6 5. Bxf6 Qxf6"
pgn_fbv10 = "1.e4 e5 2.Nf3 Nc6 3.Bc4 Nf6 4.O-O d6 4.d3 Bg4 6.h3 Bxf3 7.Qxf3"
pgn_fbv11 = "1.e4 c6 2.d4 d5 3.e5 c5 4.Nf3 Bg4 5.h3 Bxf3 6.Qxf3 cxd4"

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
                    move = infer_move.record_move(
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

def test_vision_system(test_cases):
    """
    Test the chess vision system against known PGNs.
    
    Args:
        test_cases: A list of tuples (video_path, expected_pgn)
        
    Returns:
        Dictionary containing test results
    """
    import time
    import scripts.process_video as pv
    import scripts.create_PGN
    import re
    
    results = {
        'overall': {'total_runs': 0, 'successful_runs': 0, 'accuracy': 0},
        'by_video': {},
        'by_move': {},
        'detailed': []
    }
    
    # Number of test runs per video
    runs_per_video = 10
    
    # For each test case
    for video_path, expected_pgn in test_cases:
        video_name = video_path.split('/')[-1].split('.')[0]
        results['by_video'][video_name] = {
            'total_runs': 0,
            'successful_runs': 0,
            'move_accuracy': [],
            'average_move_accuracy': 0,
            'full_pgn_accuracy': 0
        }
        
        # Parse expected PGN into individual moves
        expected_moves = parse_pgn(expected_pgn)
        
        # Setup move tracking across all runs
        results['by_move'][video_name] = {}
        for i, move in enumerate(expected_moves):
            move_id = f"Move {i+1}: {move}"
            results['by_move'][video_name][move_id] = {
                'correct': 0,
                'total': 0,
                'accuracy': 0
            }
        
        print(f"\n{'='*80}")
        print(f"TESTING VIDEO: {video_name}")
        print(f"Expected PGN: {expected_pgn}")
        print(f"{'='*80}")
        
        # Run multiple tests for this video
        for run in range(runs_per_video):
            print(f"\nRun {run+1}/{runs_per_video} for {video_name}")
            run_start_time = time.time()
            
            try:
                # Extract frames
                num_frames, frames = pv.extract_frames(
                    video_path=video_path,
                    output_dir="extracted_frames",
                    frames_interval=1
                )
                print(f'Extracted {num_frames} frames')
                
                # Process frames
                board_states, moves = process_images(frames)
                
                # Post-process moves
                post_processed_moves = post_process_moves(moves)
                
                # Generate PGN
                generated_pgn = scripts.create_PGN.create_pgn(post_processed_moves)
                
                # Parse generated PGN into individual moves
                generated_moves = parse_pgn(generated_pgn)
                
                # Calculate move-by-move accuracy
                move_accuracy = compare_moves(expected_moves, generated_moves)
                
                # Determine if full PGN matches (exact or fuzzy)
                full_pgn_match = is_pgn_match(expected_pgn, generated_pgn)
                
                # Update individual move statistics
                for i, (expected, detected, correct) in enumerate(move_accuracy):
                    move_id = f"Move {i+1}: {expected}"
                    if move_id in results['by_move'][video_name]:
                        results['by_move'][video_name][move_id]['total'] += 1
                        if correct:
                            results['by_move'][video_name][move_id]['correct'] += 1
                
                # Calculate average move accuracy for this run
                correct_moves = sum(1 for _, _, correct in move_accuracy if correct)
                total_moves = len(expected_moves)
                run_move_accuracy = correct_moves / total_moves if total_moves > 0 else 0
                
                # Update results
                results['by_video'][video_name]['total_runs'] += 1
                results['by_video'][video_name]['move_accuracy'].append(run_move_accuracy)
                if full_pgn_match:
                    results['by_video'][video_name]['successful_runs'] += 1
                
                # Add detailed results for this run
                run_time = time.time() - run_start_time
                results['detailed'].append({
                    'video': video_name,
                    'run': run + 1,
                    'expected_pgn': expected_pgn,
                    'generated_pgn': generated_pgn,
                    'move_accuracy': run_move_accuracy,
                    'full_pgn_match': full_pgn_match,
                    'processing_time': run_time,
                    'move_details': move_accuracy
                })
                
                print(f"Generated PGN: {generated_pgn}")
                print(f"Move accuracy: {run_move_accuracy:.2%} ({correct_moves}/{total_moves} moves correct)")
                print(f"Full PGN match: {'Yes' if full_pgn_match else 'No'}")
                print(f"Run completed in {run_time:.2f} seconds")
                
            except Exception as e:
                print(f"Error during run {run+1} for {video_name}: {e}")
                # Add failed run to results
                results['detailed'].append({
                    'video': video_name,
                    'run': run + 1,
                    'error': str(e)
                })
        
        # Calculate aggregate statistics for this video
        video_results = results['by_video'][video_name]
        video_results['average_move_accuracy'] = sum(video_results['move_accuracy']) / len(video_results['move_accuracy']) if video_results['move_accuracy'] else 0
        video_results['full_pgn_accuracy'] = video_results['successful_runs'] / video_results['total_runs'] if video_results['total_runs'] > 0 else 0
        
        # Update move-specific accuracy
        for move_id, move_stats in results['by_move'][video_name].items():
            move_stats['accuracy'] = move_stats['correct'] / move_stats['total'] if move_stats['total'] > 0 else 0
        
        print(f"\nResults for {video_name}:")
        print(f"Average move accuracy: {video_results['average_move_accuracy']:.2%}")
        print(f"Full PGN accuracy: {video_results['full_pgn_accuracy']:.2%} ({video_results['successful_runs']}/{video_results['total_runs']} runs matched)")
    
    # Calculate overall statistics
    total_runs = sum(video_results['total_runs'] for video_results in results['by_video'].values())
    successful_runs = sum(video_results['successful_runs'] for video_results in results['by_video'].values())
    results['overall']['total_runs'] = total_runs
    results['overall']['successful_runs'] = successful_runs
    results['overall']['accuracy'] = successful_runs / total_runs if total_runs > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"OVERALL RESULTS:")
    print(f"Total accuracy: {results['overall']['accuracy']:.2%} ({successful_runs}/{total_runs} runs successful)")
    print(f"{'='*80}")
    
    return results


def parse_pgn(pgn):
    """
    Parse PGN string into a list of individual moves.
    
    Args:
        pgn: PGN string (e.g., "1.e4 e5 2.Nf3 Nc6")
        
    Returns:
        List of moves without move numbers
    """
    # Remove move numbers and split into individual moves
    moves = []
    pgn = pgn.strip()
    
    # Replace move numbers with spaces
    pgn = re.sub(r'\d+\.', ' ', pgn)
    
    # Split by spaces and filter out empty strings
    for move in pgn.split():
        if move.strip():
            moves.append(move.strip())
    
    return moves


def compare_moves(expected_moves, generated_moves):
    """
    Compare expected moves with generated moves.
    
    Args:
        expected_moves: List of expected moves
        generated_moves: List of generated moves
        
    Returns:
        List of tuples (expected_move, detected_move, is_correct)
    """
    results = []
    
    # Convert both lists to lowercase for case-insensitive comparison
    expected_lower = [m.lower() for m in expected_moves]
    generated_lower = [m.lower() for m in generated_moves]
    
    # Compare moves
    for i, expected in enumerate(expected_moves):
        if i < len(generated_moves):
            # We have a corresponding generated move
            generated = generated_moves[i]
            is_correct = expected_lower[i] == generated_lower[i]
            results.append((expected, generated, is_correct))
        else:
            # Missing move in generated PGN
            results.append((expected, None, False))
    
    # Add any extra moves in generated PGN
    for i in range(len(expected_moves), len(generated_moves)):
        results.append((None, generated_moves[i], False))
    
    return results


def is_pgn_match(expected_pgn, generated_pgn):
    """
    Check if generated PGN matches expected PGN.
    Allow for minor formatting differences but require same moves.
    
    Args:
        expected_pgn: Expected PGN string
        generated_pgn: Generated PGN string
        
    Returns:
        Boolean indicating if PGNs match
    """
    # Parse both PGNs into move lists
    expected_moves = parse_pgn(expected_pgn)
    generated_moves = parse_pgn(generated_pgn)
    
    # Check if move lists are identical (case-insensitive)
    if len(expected_moves) != len(generated_moves):
        return False
    
    for i, expected in enumerate(expected_moves):
        if expected.lower() != generated_moves[i].lower():
            return False
    
    return True


def main():
    # Define test cases
    test_cases = [
        #("test_videos/test6.mp4", "1.d4 d5 2.Bf4 Nf6 3.e6 Bf5 4.c3 e6 5.Nf3 Bd6 6.Bxd6 Qxd6 7.Nh4 Bg6 8.Nxg6 hxg6 9.Bf3 c6 10.Nd2 Nd7 11.O-O O-O-O 12.f4 Ng4 13.Qxg4 f5 14.Qd1 Rh7 15.a4 Rh8 16.Nf3 Nf6 17.b4 Ng4 18.h3 Nf6 19.Qb3 Nh5 20.b5 Ng3 21.Rb1 Kb8 22.bxc6 Qxc6"),
        #("test_videos/test8.mp4", "1.d4 e5 2.dxe5 Nc6 3.Nf3 Qe7 4.Bf4 Qb4 5.Bd2 Qxb2 6.Nc3 Bb4 7.Rb1 Qa3 8.Rb3 Qa5 9.e3 Nxe5 10.Nxe5 Qxe5 11.Rxb4")
        ("test_videos/FBV10.mp4", "1.e4 e5 2.Nf3 Nc6 3.Bc4 Nf6 4.O-O d6 5.d3 Bg4 6.h3 Bxf3 7.Qxf3")
    ]
    
    # Run the tests
    results = test_vision_system(test_cases)
    
    # Export results to a JSON file
    import json
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"test_results_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to test_results2_{timestamp}.json")


if __name__ == "__main__":
    main()
