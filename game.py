
# Mahny Barazandehtar -20210702004 

import copy
import tkinter as tk

BOARD_SIZE = 7
EMPTY = '.'
AI_PIECE = 'A'
HUMAN_PIECE = 'H'
MAX_DEPTH = 2

class GameState:
    def __init__(self, board=None, current_player=AI_PIECE, move_count=0):
        if board is None:
            self.board = self.initial_board()
        else:
            self.board = board
        self.current_player = current_player
        self.move_count = move_count

    def initial_board(self):
        board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        # AI pieces
        board[0][0] = AI_PIECE
        board[2][0] = AI_PIECE
        board[4][6] = AI_PIECE
        board[6][6] = AI_PIECE
        
        # Human pieces
        board[0][6] = HUMAN_PIECE
        board[2][6] = HUMAN_PIECE
        board[4][0] = HUMAN_PIECE
        board[6][0] = HUMAN_PIECE
        return board

    def get_pieces(self, player):
        return [(r,c)
                for r in range(BOARD_SIZE)
                for c in range(BOARD_SIZE)
                if self.board[r][c] == player]

    def is_on_board(self, r, c):
        return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE

    def get_possible_moves(self):
        player = self.current_player
        pieces = self.get_pieces(player)
        directions = [(-1,0),(1,0),(0,-1),(0,1)]

        if len(pieces) == 0:
            return []
        
        if len(pieces) == 1:
            # Only one move needed
            moves = []
            (r1,c1) = pieces[0]
            for dr,dc in directions:
                nr,nc = r1+dr,c1+dc
                if self.is_on_board(nr,nc) and self.board[nr][nc] == EMPTY:
                    moves.append([((r1,c1),(nr,nc))])
            return moves
        else:
            # Two moves needed with two distinct pieces
            first_moves = []
            for i, (r1,c1) in enumerate(pieces):
                for dr,dc in directions:
                    nr1,nc1 = r1+dr,c1+dc
                    if self.is_on_board(nr1,nc1) and self.board[nr1][nc1] == EMPTY:
                        first_moves.append(((r1,c1),(nr1,nc1), i))

            moves = []
            for (rFrom1,cFrom1),(rTo1,cTo1), piece_index_1 in first_moves:
                temp_board = copy.deepcopy(self.board)
                temp_board[rFrom1][cFrom1] = EMPTY
                temp_board[rTo1][cTo1] = player

                pieces_after = [(rr,cc)
                                for rr in range(BOARD_SIZE)
                                for cc in range(BOARD_SIZE)
                                if temp_board[rr][cc] == player]

                second_moves = []
                for (r2,c2) in pieces_after:
                    # Must move a different piece:
                    if (r2,c2) == (rTo1,cTo1):
                        continue
                    for dr2,dc2 in directions:
                        nr2,nc2 = r2+dr2,c2+dc2
                        if 0 <= nr2 < BOARD_SIZE and 0 <= nc2 < BOARD_SIZE:
                            if temp_board[nr2][nc2] == EMPTY:
                                second_moves.append(((r2,c2),(nr2,nc2)))

                for (rFrom2,cFrom2),(rTo2,cTo2) in second_moves:
                    moves.append([((rFrom1,cFrom1),(rTo1,cTo1)),
                                  ((rFrom2,cFrom2),(rTo2,cTo2))])
            return moves

    def make_move(self, move_sequence):
        current_opponent = (HUMAN_PIECE if self.current_player == AI_PIECE
                            else AI_PIECE)
        opponent_before = len(self.get_pieces(current_opponent))

        new_board = copy.deepcopy(self.board)
        for (rFrom,cFrom),(rTo,cTo) in move_sequence:
            piece = new_board[rFrom][cFrom]
            new_board[rFrom][cFrom] = EMPTY
            new_board[rTo][cTo] = piece

        new_board = capture_pieces(new_board)
        next_player = (AI_PIECE if self.current_player == HUMAN_PIECE
                       else HUMAN_PIECE)
        new_state = GameState(new_board, next_player, self.move_count+1)

        opponent_after = len(new_state.get_pieces(current_opponent))
        captured = opponent_before - opponent_after
        return new_state, captured

    def is_terminal(self):
        ai_pieces = len(self.get_pieces(AI_PIECE))
        human_pieces = len(self.get_pieces(HUMAN_PIECE))
        
        if ai_pieces == 0 and human_pieces == 0:
            return True
        if ai_pieces == 1 and human_pieces == 1:
            return True
        if ai_pieces == 0 and human_pieces > 0:
            return True
        if human_pieces == 0 and ai_pieces > 0:
            return True
        if self.move_count >= 50:
            return True

        if len(self.get_possible_moves()) == 0:
            return True

        return False

    def game_result(self):
        ai_pieces = len(self.get_pieces(AI_PIECE))
        human_pieces = len(self.get_pieces(HUMAN_PIECE))

        if ai_pieces == 0 and human_pieces == 0:
            return 0
        if ai_pieces == 1 and human_pieces == 1:
            return 0
        if ai_pieces > 0 and human_pieces == 0:
            return 1
        if ai_pieces == 0 and human_pieces > 0:
            return -1
        if self.move_count >= 50:
            if ai_pieces == human_pieces:
                return 0
            elif ai_pieces > human_pieces:
                return 1
            else:
                return -1

        if len(self.get_possible_moves()) == 0:
            # If current player can't move, they lose
            if self.current_player == AI_PIECE:
                return -1
            else:
                return 1

        return None

    def evaluate(self):
        ai_pieces = len(self.get_pieces(AI_PIECE))
        human_pieces = len(self.get_pieces(HUMAN_PIECE))
        piece_score = (ai_pieces - human_pieces) * 10

        current_player_saved = self.current_player
        self.current_player = AI_PIECE
        ai_moves = len(self.get_possible_moves())
        self.current_player = HUMAN_PIECE
        human_moves = len(self.get_possible_moves())
        self.current_player = current_player_saved

        mobility_score = (ai_moves - human_moves) * 2

        potential_score = 0
        if not self.is_terminal():
            saved_player = self.current_player
            self.current_player = AI_PIECE
            before = len(self.get_pieces(HUMAN_PIECE))
            best_capture = 0
            for move in self.get_possible_moves():
                temp_state, _ = self.make_move(move)
                after = len(temp_state.get_pieces(HUMAN_PIECE))
                diff = before - after
                if diff > best_capture:
                    best_capture = diff
            potential_score += best_capture * 5
            self.current_player = saved_player

        return piece_score + mobility_score + potential_score


def alphabeta(state, depth, alpha, beta, maximizing_player):
    if depth == 0 or state.is_terminal():
        return state.evaluate(), None

    moves = state.get_possible_moves()
    if not moves:
        return state.evaluate(), None

    if maximizing_player:
        best_val = float('-inf')
        best_move = None
        for move in moves:
            new_state, _ = state.make_move(move)
            val, _ = alphabeta(new_state, depth-1, alpha, beta, False)
            if val > best_val:
                best_val = val
                best_move = move
            alpha = max(alpha, best_val)
            if beta <= alpha:
                break
        return best_val, best_move
    else:
        best_val = float('inf')
        best_move = None
        for move in moves:
            new_state, _ = state.make_move(move)
            val, _ = alphabeta(new_state, depth-1, alpha, beta, True)
            if val < best_val:
                best_val = val
                best_move = move
            beta = min(beta, best_val)
            if beta <= alpha:
                break
        return best_val, best_move


def check_line(line):
    # If left flank is None, only treat it as "wall" if start == 0
    # If right flank is None, only treat it as "wall" if end == len(line) - 1
    changed_line = list(line)
    length = len(changed_line)
    start = 0

    while start < length:
        if changed_line[start] != EMPTY:
            piece_type = changed_line[start]
            end = start
            while end + 1 < length and changed_line[end + 1] == piece_type:
                end += 1

            left_index = start - 1
            right_index = end + 1

            left_piece = changed_line[left_index] if left_index >= 0 else None
            right_piece = changed_line[right_index] if right_index < length else None

            def is_enemy_or_wall(flank, current_type, is_left_side):
                if flank is None:
                    # Only treat this as a "wall" pinch if we're truly at the edge with no gap.  i.e. left_index < 0 => start == 0 (for left),or right_index >= length => end == length-1 (for right).
                    if is_left_side and start == 0:
                        return True
                    if not is_left_side and end == length - 1:
                        return True
                    return False
                if flank == EMPTY:
                    return False  # empty cell breaks the pinch
                return flank != current_type  # enemy

            left_flank = is_enemy_or_wall(left_piece, piece_type, is_left_side=True)
            right_flank = is_enemy_or_wall(right_piece, piece_type, is_left_side=False)

            if left_flank and right_flank:
                for i in range(start, end + 1):
                    changed_line[i] = EMPTY

            start = end + 1
        else:
            start += 1

    return changed_line

def capture_pieces(board):
    while True:
        changed = False

        # Process rows
        for r in range(BOARD_SIZE):
            new_row = check_line(board[r])
            if new_row != board[r]:
                board[r] = new_row
                changed = True

        # Process columns
        for c in range(BOARD_SIZE):
            col = [board[r][c] for r in range(BOARD_SIZE)]
            new_col = check_line(col)
            if new_col != col:
                for r in range(BOARD_SIZE):
                    board[r][c] = new_col[r]
                changed = True

        if not changed:
            break

    return board


class GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Mahny Barazandehtar - 20210702004")
        self.cell_size = 60
        
        self.state = GameState()
        
        # For partial moves display
        self.turn_start_board = None
        self.display_board = copy.deepcopy(self.state.board)

        self.frame = tk.Frame(root)
        self.frame.pack(side=tk.TOP)
        self.turn_label = tk.Label(self.frame, text="", font=("Arial", 14))
        self.turn_label.pack()

        self.canvas = tk.Canvas(
            root, width=self.cell_size*BOARD_SIZE, height=self.cell_size*BOARD_SIZE
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)

        # Track how many moves remain in the human's current turn
        self.moves_left_for_turn = 0
        self.partial_move_sequence = []

        # For highlighting possible moves
        self.possible_moves = []

        self.selected_piece = None
        self.game_over = False

        self.draw_board()
        self.update_turn_label()
        self.check_end_game()

        # If AI goes first:
        if not self.game_over and self.state.current_player == AI_PIECE:
            self.ai_move()

    def update_turn_label(self, text=None, color="black"):
        if self.game_over:
            return
        if text is not None:
            self.turn_label.config(text=text, fg=color)
        else:
            if self.state.current_player == AI_PIECE:
                self.turn_label.config(text="AI's Turn", fg="red")
            else:
                self.turn_label.config(text="Human's Turn", fg="blue")

    def draw_board(self):
        self.canvas.delete("all")

        # Draw squares
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                x1 = c*self.cell_size
                y1 = r*self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                self.canvas.create_rectangle(x1, y1, x2, y2, fill="white")

        # Draw pieces from the display board
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                piece = self.display_board[r][c]
                x1 = c*self.cell_size
                y1 = r*self.cell_size
                if piece == AI_PIECE:
                    # AI piece (triangle in red)
                    self.canvas.create_polygon(x1+30,y1+10, x1+10,y1+50,
                                               x1+50,y1+50, fill="red")
                elif piece == HUMAN_PIECE:
                    # Human piece (circle in blue)
                    self.canvas.create_oval(x1+10, y1+10, x1+50, y1+50,
                                            fill="blue")

        # Highlight the selected piece
        if self.selected_piece is not None:
            (sr, sc) = self.selected_piece
            sx1 = sc*self.cell_size
            sy1 = sr*self.cell_size
            sx2 = sx1 + self.cell_size
            sy2 = sy1 + self.cell_size
            self.canvas.create_rectangle(sx1, sy1, sx2, sy2,
                                         fill="yellow", stipple="gray50")

        # Highlight possible moves
        for (mr, mc) in self.possible_moves:
            center_x = mc * self.cell_size + self.cell_size // 2
            center_y = mr * self.cell_size + self.cell_size // 2
            radius = 10
            # Draw a small green circle
            self.canvas.create_oval(center_x - radius, center_y - radius,
                                    center_x + radius, center_y + radius,
                                    fill="green", outline="")

    def get_possible_moves_for_piece_display(self, r, c):
        if self.display_board[r][c] not in (AI_PIECE, HUMAN_PIECE):
            return []
        directions = [(-1,0), (1,0), (0,-1), (0,1)]
        valid_positions = []
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                if self.display_board[nr][nc] == EMPTY:
                    valid_positions.append((nr, nc))
        return valid_positions

    def on_click(self, event):
        if self.game_over:
            return
        if self.state.current_player == AI_PIECE:
            return  # Humans only move during their turn

        c = event.x // self.cell_size
        r = event.y // self.cell_size
        if not self.state.is_on_board(r, c):
            return

        # If we haven't set up moves_left_for_turn yet, do so:
        if self.moves_left_for_turn == 0:
            self.turn_start_board = copy.deepcopy(self.state.board)
            self.display_board = copy.deepcopy(self.turn_start_board)

            human_piece_count = len(self.state.get_pieces(HUMAN_PIECE))
            self.moves_left_for_turn = 1 if human_piece_count <= 1 else 2
            self.partial_move_sequence = []

        # Disallow re-selecting the same piece
        # If we already made 1 partial move and have 1 left, record which piece was moved:
        # We'll store the final position of the piece that was just moved
        already_moved_piece = None
        if len(self.partial_move_sequence) == 1:
            # The piece that moved first ended up at partial_move_sequence[0][1]
            # e.g. ((r1,c1),(r2,c2))
            already_moved_piece = self.partial_move_sequence[0][1]

        if self.selected_piece is None:
            # Pick a piece if valid
            if self.display_board[r][c] == HUMAN_PIECE:
                # If there's a piece that already moved, skip if user re-selects it
                if (already_moved_piece is not None) and (r, c) == already_moved_piece:
                    self.show_temporary_message("Can't move the same piece twice!", "orange", 1500)
                    return
                # Otherwise, select it
                self.selected_piece = (r, c)
                self.possible_moves = self.get_possible_moves_for_piece_display(r, c)
                self.draw_board()
            else:
                return
        else:
            start = self.selected_piece
            end = (r, c)

            # If user clicked another piece (instead of a valid empty square)
            if self.display_board[r][c] == HUMAN_PIECE and (r,c) != start:
                # Check if it's the piece that already moved
                if (already_moved_piece is not None) and (r, c) == already_moved_piece:
                    self.show_temporary_message("Can't move the same piece twice!", "orange", 1500)
                    return
                # Switch selection to the newly clicked piece
                self.selected_piece = (r, c)
                self.possible_moves = self.get_possible_moves_for_piece_display(r, c)
                self.draw_board()
                return

            # Otherwise, see if we clicked a valid destination
            if end in self.possible_moves:
                self.apply_partial_move(start, end)
            else:
                # Invalid
                self.selected_piece = None
                self.possible_moves = []
                self.draw_board()
                self.show_temporary_message("Invalid Move! Try Again.", "orange", 1000)

    def apply_partial_move(self, start, end):
        piece = self.display_board[start[0]][start[1]]

        # Move on the display board
        self.display_board[start[0]][start[1]] = EMPTY
        self.display_board[end[0]][end[1]] = piece

        # Append partial move
        self.partial_move_sequence.append((start, end))
        self.moves_left_for_turn -= 1

        # Clear selection
        self.selected_piece = None
        self.possible_moves = []
        self.draw_board()

        if self.moves_left_for_turn > 0:
            self.show_temporary_message("One more move left!", "blue", 1500)
        else:
            self.execute_human_turn()

    def execute_human_turn(self):
        # Revert real board to turn_start_board
        self.state.board = copy.deepcopy(self.turn_start_board)

        # Apply partial moves fully
        new_state, captured = self.state.make_move(self.partial_move_sequence)
        if captured > 0:
            self.show_capture_message(HUMAN_PIECE, captured)

        self.state = new_state
        self.display_board = copy.deepcopy(self.state.board)

        self.partial_move_sequence = []
        self.moves_left_for_turn = 0

        self.draw_board()
        self.check_end_game()

        if not self.game_over:
            self.update_turn_label()
            if self.state.current_player == AI_PIECE:
                self.ai_move()

    def ai_move(self):
        if self.game_over:
            return
        self.root.after(500, self.perform_ai_move)

    def perform_ai_move(self):
        if self.game_over:
            return
        val, best_move = alphabeta(
            self.state,
            MAX_DEPTH,
            float('-inf'),
            float('inf'),
            self.state.current_player == AI_PIECE
        )
        if best_move:
            new_state, captured = self.state.make_move(best_move)
            if captured > 0:
                self.show_capture_message(AI_PIECE, captured)
            self.state = new_state

        self.display_board = copy.deepcopy(self.state.board)
        self.draw_board()
        self.check_end_game()
        if not self.game_over:
            self.update_turn_label()

    def show_temporary_message(self, message, color, duration_ms=1000):
        current_text = self.turn_label.cget("text")
        current_fg = self.turn_label.cget("fg")
        self.turn_label.config(text=message, fg=color)
        self.root.after(duration_ms, lambda: self.update_turn_label())

    def show_capture_message(self, player, num_captured):
        if num_captured > 0:
            player_name = "AI (A)" if player == AI_PIECE else "Human (H)"
            msg = f"{player_name} captured {num_captured} piece(s)!"
            self.show_temporary_message(msg, "green", 1500)

    def check_end_game(self):
        if self.game_over:
            return
        if self.state.is_terminal():
            self.game_over = True
            result = self.state.game_result()
            if result == 1:
                end_text = "Game Over: AI wins!"
            elif result == -1:
                end_text = "Game Over: Human wins!"
            else:
                end_text = "Game Over: It's a draw!"
            self.turn_label.config(text=end_text, fg="green")


if __name__ == "__main__":
    root = tk.Tk()
    gui = GUI(root)
    root.mainloop()
