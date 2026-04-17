
class Board:

  state; #8x8 grid with pieces
  
  class Piece:
    __init__(self):
      #arbitrary state variables (counters, etc)

    on_board_update(self, Board.state):
      #hook that gets called on every move: updates state vars, board state if necessary

    available_moves(self, Board.state):
      #return an an array of possible future board states based on current state
      #this basically describes a piece
    
  __init__(self):
    #initializes board state with random magical pieces etc