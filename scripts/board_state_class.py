class BoardState:
    def __init__(self, state, occupied_squares):
        self.state = state
        self.occupied_squares = occupied_squares
    
    def __str__(self):
        """Returns a string representation of the board state"""
        return f"Occupied Squares: {', '.join(self.occupied_squares)}\nBoard State:\n{self.state}"