

def is_simple_move(board_changes):
    print('CHECKING IF IS SIMPLE MOVE')
    print(len(board_changes['newly_empty']))
    print(len(board_changes['newly_occupied']))
    print(len(board_changes['color_changed']))
    """Check if the changes represent a simple move (one piece moving to an empty square)"""
    return (len(board_changes['newly_empty']) == 1 and 
            len(board_changes['newly_occupied']) == 1 and 
            len(board_changes['color_changed']) == 0)


def handle_simple_move(board_changes, current_board_state):
    print('entered handle simple ')
    """Handle a simple move and update the board state"""
    source_square = next(iter(board_changes['newly_empty']))
    destination_square = next(iter(board_changes['newly_occupied']))
    
    # Get the piece information from the source square
    moving_piece = board_changes['previous_board_state'][source_square]
    
    # Update the board state: remove piece from source square
    current_board_state[source_square] = None
    
    # Add the moved piece to its new location
    current_board_state[destination_square] = moving_piece
    
    print(f"Simple move detected: {source_square} → {destination_square}")
    print(f"Piece moved: {moving_piece['piece']} ({moving_piece['color']})")
    
    return current_board_state, source_square, destination_square


def is_capture_move(board_changes):
    """Check if the changes represent a capture move"""
    return (len(board_changes['newly_empty']) == 1 and 
            len(board_changes['color_changed']) == 1)


def handle_capture_move(board_changes, current_board_state):
    """Handle a capture move and update the board state"""
    source_square = next(iter(board_changes['newly_empty']))
    destination_square = board_changes['color_changed'][0]
    
    # Get pieces involved
    moving_piece = board_changes['previous_board_state'][source_square]
    captured_piece = board_changes['previous_board_state'][destination_square]
    
    # Update the board state
    current_board_state[source_square] = None
    current_board_state[destination_square] = moving_piece
    
    print(f"Capture detected: {source_square} → {destination_square}")
    print(f"Piece moved: {moving_piece['piece']} ({moving_piece['color']})")
    print(f"Piece captured: {captured_piece['piece']} ({captured_piece['color']})")
    
    return current_board_state, source_square, destination_square, captured_piece


def is_castling_move(board_changes):
    """
    Check if the changes represent a castling move.
    
    In castling, both the king and rook move in a specific pattern:
    1. The king moves two squares towards the rook
    2. The rook moves to the square the king crossed
    
    We detect this by looking for specific square patterns that got newly occupied.
    """
    # For castling, we typically see two squares become empty and two become occupied
    newly_empty = board_changes['newly_empty']
    newly_occupied = board_changes['newly_occupied']

    print("ENTERED CHECK CASTLE")
    print("NEWLY EMPTY")
    print(len(newly_empty))
    print("NEWLY OCCUPIED")
    print(len(newly_occupied))
    
    # Short castling (kingside) squares: g1,f1 for White or g8,f8 for Black
    # Long castling (queenside) squares: c1,d1 for White or c8,d8 for Black
    white_kingside_squares = {'g1', 'f1'}
    white_queenside_squares = {'c1', 'd1'}
    black_kingside_squares = {'g8', 'f8'}
    black_queenside_squares = {'c8', 'd8'}
    
    # Check if the newly occupied squares match any of the castling patterns
    if len(newly_empty) == 2 and len(newly_occupied) == 2:
        newly_occupied_set = set(newly_occupied)
        
        # Check for White castling
        if newly_occupied_set == white_kingside_squares:
            return {'castling_type': 'white_kingside'}
        elif newly_occupied_set == white_queenside_squares:
            return {'castling_type': 'white_queenside'}
        
        # Check for Black castling
        elif newly_occupied_set == black_kingside_squares:
            return {'castling_type': 'black_kingside'}
        elif newly_occupied_set == black_queenside_squares:
            return {'castling_type': 'black_queenside'}
    
    # No castling pattern found
    return False


def handle_castling_move(board_changes, current_board_state):
    """
    Handle a castling move and update the board state.
    
    This function updates the positions of both king and rook based on the type of castling.
    """
    # Get castling information
    castling_info = is_castling_move(board_changes)
    castling_type = castling_info['castling_type']
    
    # Define source and destination squares for king and rook based on castling type
    if castling_type == 'white_kingside':
        king_source = 'e1'
        king_dest = 'g1'
        rook_source = 'h1'
        rook_dest = 'f1'
        
    elif castling_type == 'white_queenside':
        king_source = 'e1'
        king_dest = 'c1'
        rook_source = 'a1'
        rook_dest = 'd1'
        
    elif castling_type == 'black_kingside':
        king_source = 'e8'
        king_dest = 'g8'
        rook_source = 'h8'
        rook_dest = 'f8'
        
    elif castling_type == 'black_queenside':
        king_source = 'e8'
        king_dest = 'c8'
        rook_source = 'a8'
        rook_dest = 'd8'
    
    # Get the king and rook pieces
    king_piece = board_changes['previous_board_state'][king_source]
    rook_piece = board_changes['previous_board_state'][rook_source]
    
    # Update the board state: move king and rook to their new positions
    current_board_state[king_source] = None
    current_board_state[rook_source] = None
    current_board_state[king_dest] = king_piece
    current_board_state[rook_dest] = rook_piece
    
    # For the PGN notation, we consider the king's move as the main move
    print(f"Castling detected: {castling_type}")
    print(f"King moved: {king_source} → {king_dest}")
    print(f"Rook moved: {rook_source} → {rook_dest}")
    
    # Return king's move information for PGN notation
    return current_board_state, king_source, king_dest, castling_type


def is_en_passant_move(board_changes):
    """Check if the changes represent an en passant capture"""
    # En passant has a specific pattern:
    # - Pawn moves diagonally
    # - The captured pawn disappears from a different square than the destination
    # Implementation to be added
    return False


def handle_en_passant_move(board_changes, current_board_state):
    """Handle an en passant move and update the board state"""
    # Implementation to be added
    return current_board_state, None, None, None


def is_promotion_move(board_changes):
    """Check if the changes represent a pawn promotion"""
    # A pawn reaches the last rank and transforms into another piece
    # Implementation to be added
    return False


def handle_promotion_move(board_changes, current_board_state):
    """Handle a promotion move and update the board state"""
    # Implementation to be added
    return current_board_state, None, None, False, None

def record_move(source_square, destination_square, previous_board_state, is_capture, captured_piece, is_castle, castle_type):
    """
    Create a move dictionary based on detected board changes.
    
    Parameters:
        source_square: Starting square of the move
        destination_square: Ending square of the move
        previous_board_state: Board state before the move
        is_capture: Boolean indicating if the move was a capture
        captured_piece: Information about captured piece (if any)
        is_castle: Boolean indicating if the move was castling
        castle_type: Type of castling (if applicable)
        
    Returns:
        Dictionary representing the chess move
    """
    if not is_castle:
        print('Normal or capture move detected')
        # Create a move object for normal moves and captures
        move = {
            'from': source_square,
            'to': destination_square,
            'piece': previous_board_state[source_square]['piece'] if source_square in previous_board_state else 'Unknown',
            'color': previous_board_state[source_square]['color'] if source_square in previous_board_state else 'Unknown',
            'is_capture': is_capture
        }
        
        # Add captured piece information if this was a capture
        if is_capture and captured_piece:
            move['captured_piece'] = captured_piece['piece']
            move['captured_color'] = captured_piece['color']
    else:
        print('Castling move detected')
        # Create a move object for castling
        move = {
            'from': source_square,
            'to': destination_square,
            'piece': previous_board_state[source_square]['piece'] if source_square in previous_board_state else 'Unknown',
            'color': previous_board_state[source_square]['color'] if source_square in previous_board_state else 'Unknown',
            'is_castle': True
        }
        
        # Determine castle type
        if castle_type == 'white_kingside' or castle_type == 'black_kingside':
            move['castle_type'] = 'short'
        elif castle_type == 'white_queenside' or castle_type == 'black_queenside':
            move['castle_type'] = 'long'
    
    print('The move is:', move)
    return move