import scripts.infer_move


def construct_initial_board_state(occupied_squares_with_color=None):
    """
    Construct the initial chess board state based on the standard starting position.
    
    This function creates a complete representation of a chess board's starting position,
    including both occupied and unoccupied squares. Each square is represented with
    information about its piece (if any) and the color of that piece.
    
    Args:
        occupied_squares_with_color: Optional list of tuples (square_name, color) for validation.
            If provided, this will be used to validate that the detected pieces match
            the expected starting position.
    
    Returns:
        board_state: Dictionary where:
            - Keys are chess coordinates (e.g., 'e4')
            - Values are dictionaries with keys 'piece' and 'color', or None for empty squares
              Example: {'piece': 'P', 'color': 'white'} for a white pawn
    """
    # Create an empty board
    board_state = {}
    for file in 'abcdefgh':
        for rank in range(1, 9):
            square = f"{file}{rank}"
            board_state[square] = None  # Initialize all squares as empty
    
    # Set up the standard starting position
    # White pieces (bottom row)
    board_state['a1'] = {'piece': 'R', 'color': 'white'}  # Rook
    board_state['b1'] = {'piece': 'N', 'color': 'white'}  # Knight
    board_state['c1'] = {'piece': 'B', 'color': 'white'}  # Bishop
    board_state['d1'] = {'piece': 'Q', 'color': 'white'}  # Queen
    board_state['e1'] = {'piece': 'K', 'color': 'white'}  # King
    board_state['f1'] = {'piece': 'B', 'color': 'white'}  # Bishop
    board_state['g1'] = {'piece': 'N', 'color': 'white'}  # Knight
    board_state['h1'] = {'piece': 'R', 'color': 'white'}  # Rook
    
    # White pawns
    for file in 'abcdefgh':
        square = f"{file}2"
        board_state[square] = {'piece': 'P', 'color': 'white'}
    
    # Black pieces (top row)
    board_state['a8'] = {'piece': 'r', 'color': 'black'}  # Rook
    board_state['b8'] = {'piece': 'n', 'color': 'black'}  # Knight
    board_state['c8'] = {'piece': 'b', 'color': 'black'}  # Bishop
    board_state['d8'] = {'piece': 'q', 'color': 'black'}  # Queen
    board_state['e8'] = {'piece': 'k', 'color': 'black'}  # King
    board_state['f8'] = {'piece': 'b', 'color': 'black'}  # Bishop
    board_state['g8'] = {'piece': 'n', 'color': 'black'}  # Knight
    board_state['h8'] = {'piece': 'r', 'color': 'black'}  # Rook
    
    # Black pawns
    for file in 'abcdefgh':
        square = f"{file}7"
        board_state[square] = {'piece': 'p', 'color': 'black'}
    
    # If occupied squares were provided, validate that they match the expected starting position
    if occupied_squares_with_color is not None:
        # Convert detected occupied squares to a dictionary for easier lookup
        detected_occupied = {square: color for square, color in occupied_squares_with_color}
        
        # Validate occupied squares
        expected_occupied = {square: data['color'] 
                            for square, data in board_state.items() 
                            if data is not None}
        
        # Check for discrepancies
        discrepancies = []
        
        # Check for missing pieces (should be occupied but weren't detected)
        for square, expected_color in expected_occupied.items():
            if square not in detected_occupied:
                discrepancies.append(f"Missing piece: {square} should have a {expected_color} piece")
            elif detected_occupied[square] != expected_color:
                discrepancies.append(f"Color mismatch: {square} has {detected_occupied[square]} piece, expected {expected_color}")
        
        # Check for unexpected pieces (detected but shouldn't be there)
        for square, detected_color in detected_occupied.items():
            if square not in expected_occupied:
                discrepancies.append(f"Unexpected piece: {square} has a {detected_color} piece, should be empty")
        
        # Report discrepancies if any were found
        if discrepancies:
            print("Warning: Detected pieces don't match the expected starting position:")
            for discrepancy in discrepancies:
                print(f"  - {discrepancy}")
            print("Using standard starting position anyway.")
    
    return board_state

def analyze_board_changes(current_occupied_squares, previous_board_state):
    """
    Analyze changes between the current and previous board states.
    
    Args:
        current_occupied_squares: List of currently occupied squares with their colors
        previous_board_state: Dictionary of the previous board state
    
    Returns:
        Dictionary containing various sets of changed squares
    """
    # Create a mapping of current squares to their colors for easier lookup
    current_square_colors = {square: color for square, color in current_occupied_squares}
    
    # Extract just the square names for the currently occupied squares
    current_squares = set(current_square_colors.keys())
    
    # Find previously occupied squares (squares with non-None values)
    previous_occupied_squares = {square for square, piece_info in previous_board_state.items() 
                               if piece_info is not None}
    
    # Find squares that changed
    newly_empty_squares = previous_occupied_squares - current_squares
    newly_occupied_squares = current_squares - previous_occupied_squares
    
    # Identify squares that remained occupied but might have a different piece now
    still_occupied_squares = current_squares.intersection(previous_occupied_squares)
    
    # Look for squares where the color changed (indicating a capture)
    color_changed_squares = []
    for square in still_occupied_squares:
        current_color = current_square_colors[square]
        previous_color = previous_board_state[square]['color']
        
        # Compare colors (considering different possible formats)
        if current_color.lower() != previous_color.lower():
            color_changed_squares.append(square)

    print('BOARD CHANGES')
    print(f"new_empty: {newly_empty_squares}\nnewly_occu: {newly_occupied_squares}\n")
    
    return {
        'newly_empty': newly_empty_squares,
        'newly_occupied': newly_occupied_squares,
        'still_occupied': still_occupied_squares,
        'color_changed': color_changed_squares,
        'current_square_colors': current_square_colors,
        'previous_board_state': previous_board_state
    }


def create_current_board_state(current_occupied_squares, previous_board_state):
    """
    Create the current board state by comparing currently occupied squares with the previous board state.
    This allows us to track pieces as they move and maintain their identity.
    
    Args:
        current_occupied_squares: List of squares that are occupied, with format [(square_name, color), ...]
        previous_board_state: Dictionary mapping square names to piece info from the previous state
    
    Returns:
        current_board_state: Updated dictionary with the new board state
        source_square: The square a piece moved from (if a move was detected)
        destination_square: The square a piece moved to (if a move was detected)
        is_capture: Boolean indicating whether the move was a capture
        captured_piece: Information about the captured piece (if any)
    """
    # Analyze changes in the board position
    board_changes = analyze_board_changes(current_occupied_squares, previous_board_state)

    is_not_valid = False

    if (len(board_changes['newly_empty']) == 0 and 
        len(board_changes['newly_occupied']) == 0 and 
        len(board_changes['color_changed']) == 0):
        print("No changes detected - board state is unchanged from previous frame")
        return (previous_board_state, None, None, False, None, False, None, False)
    
    if len(board_changes['color_changed']) > 1:
        is_not_valid = True

    if (len(board_changes['newly_empty']) == 1 and len(board_changes['newly_occupied']) == 0 and len(board_changes['color_changed']) != 1):
        is_not_valid = True

    print(f"is valid due to color: {is_not_valid}")
    # Create a copy of the previous board state to start with
    current_board_state = dict(previous_board_state)
    
    # Initialize return values
    source_square = None
    destination_square = None
    is_capture = False
    captured_piece = None
    is_castle = False
    castle_type = None
    
    # Determine the type of move and update the board state accordingly
    if  scripts.infer_move.is_simple_move(board_changes):
        print("is simple move")
        current_board_state, source_square, destination_square = scripts.infer_move.handle_simple_move(
            board_changes, current_board_state)
            
    elif scripts.infer_move.is_capture_move(board_changes):
        print("is capture move")
        current_board_state, source_square, destination_square, captured_piece = scripts.infer_move.handle_capture_move(
            board_changes, current_board_state)
        is_capture = True
        
    elif scripts.infer_move.is_castling_move(board_changes):
        print("is castle move")
        current_board_state, source_square, destination_square, castling_type = scripts.infer_move.handle_castling_move(
            board_changes, current_board_state)
        is_castle = True
        castle_type = castling_type
        
    elif scripts.infer_move.is_en_passant_move(board_changes):
        current_board_state, source_square, destination_square, captured_piece = scripts.infer_move.handle_en_passant_move(
            board_changes, current_board_state)
        is_capture = True
        
    elif scripts.infer_move.is_promotion_move(board_changes):
        current_board_state, source_square, destination_square, is_capture, captured_piece = scripts.infer_move.handle_promotion_move(
            board_changes, current_board_state)
        
    return current_board_state, source_square, destination_square, is_capture, captured_piece, is_castle, castle_type, is_not_valid






