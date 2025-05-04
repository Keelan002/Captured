def create_pgn(moves):
    """
    Create a PGN string from a list of move dictionaries.
    
    Parameters:
        moves: List of move dictionaries with keys 'from', 'to', 'piece', 'color', 'is_capture'
    
    Returns:
        PGN string representation of the moves
    """
    pgn = ""
    move_pairs = []
    current_pair = []
    
    # Group moves into white-black pairs
    for move in moves:
        color = move['color'].lower()
        
        if color == 'white':
            # Start a new pair if we've already seen a white move
            if len(current_pair) > 0 and current_pair[0]['color'].lower() == 'white':
                move_pairs.append(current_pair)
                current_pair = []
            
            # Add this white move to the current pair
            current_pair.append(move)
        else:  # Black move
            # Add this black move to the current pair if we have a white move
            if len(current_pair) > 0:
                current_pair.append(move)
                move_pairs.append(current_pair)
                current_pair = []
            else:
                # Orphaned black move - create a pair with an empty white move
                current_pair = [None, move]
                move_pairs.append(current_pair)
                current_pair = []
    
    # Add any remaining moves
    if current_pair:
        move_pairs.append(current_pair)
    
    # Format the PGN
    for i, pair in enumerate(move_pairs):
        move_number = i + 1
        pgn += f"{move_number}. "
        
        # Format the white move
        if pair[0]:
            pgn += format_move(pair[0])
        
        # Format the black move if it exists
        if len(pair) > 1 and pair[1]:
            pgn += " " + format_move(pair[1])
        
        pgn += " "
    
    return pgn.strip()

def format_move(move):
    """Format a single move in PGN notation."""
    if not move:
        return ""
        
    from_square = move['from']
    to_square = move['to']
    piece = move['piece']
    is_capture = move.get('is_capture', False)
    is_castle = move.get('is_castle', False)
    castle_type = move.get('castle_type', None)
    
    # Handle castling
    if is_castle and castle_type:
        if castle_type == 'short':
            return "O-O"
        elif castle_type == 'long':
            return "O-O-O"
    
    # Regular move notation
    piece_letter = ""
    if piece.upper() in ['R', 'N', 'B', 'Q', 'K']:
        # Uppercase for standard notation regardless of color
        piece_letter = piece.upper()
    
    # Format the move string
    if piece.upper() == 'P':
        # Pawn moves
        if is_capture:
            # For pawn captures, use the file (column) of the starting square
            return f"{from_square[0]}x{to_square}"
        else:
            # For normal pawn moves, just use the destination square
            return f"{to_square}"
    else:
        # Piece moves
        if is_capture:
            return f"{piece_letter}x{to_square}"
        else:
            return f"{piece_letter}{to_square}"