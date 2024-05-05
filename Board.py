import numpy as np

class Board():
    """
    This class defines the board as a 2x5x5 tensor. The first layer represents the locations of the workers and the second layer
    represents the heights of the buildings. The board is initialized with two workers per player placed randomly on the board.
    Players take turns moving their workers and building buildings. The game ends when one player reaches the third level of a building
    or when a player cannot move any of their workers. The game is played in a loop until the game ends.
    """

    def __init__(self) -> None:
        """
        Initialize the board.
        """
        self.board = np.zeros((2, 5, 5))
        self.workers = np.zeros((5, 5))
        self.levels = np.zeros((5, 5))
        self.winner = 0
        self.init_board()

    def init_board(self) -> None:
        """
        Initialize the board with the workers placed randomly on the board.
        2 workers will be placed for each player with indexes 1 and 2 for player 1 and -1 and -2 for player 2.
        """
        places = np.random.choice(25, 4, replace=False)
        workers = [1, 2, -1, -2]
        for i in range(4):
            self.workers[places[i] // 5][places[i] % 5] = workers[i]
        self.board[0] = self.workers
        self.board[1] = self.levels

    def get_state(self) -> np.array:
        """
        Return the current state of the board.
        """
        return self.board

    def get_legal_moves(self, player) -> np.array:
        """
        Get the legal moves for the player.
        :param player: The player whose turn it is.
        :return: A list of legal moves for the player.
        """
        actions = []
        player_workers = [i * (-1 ** (player - 1)) for i in [1, 2]]
        worker_positions = np.argwhere(self.workers == player_workers[0])
        # for each worker, check orthogonal and diagonal moves
        for worker in worker_positions:
            for i in range(-1, 2):
                for j in range(-1, 2):
                    worker_level = self.levels[worker[0]][worker[1]]
                    cur_i, cur_j = worker[0], worker[1]
                    move_i, move_j = cur_i + i, cur_j + j
                    if (move_i >= 0 and move_i < 5 # bounds check
                        and move_j >= 0 and move_j < 5
                        and self.workers[move_i][move_j] == 0 # check if worker is already there
                        and abs(self.levels[move_i][move_j] - worker_level) <= 1 # check if worker can move to that level
                        and self.levels[move_i][move_j] < 4):
                        # for each move, check orthogonal building locations
                        for build_i in range(move_i - 1, move_i + 2):
                            for build_j in range(move_j - 1, move_j + 2):
                                if (build_i >= 0 and build_i < 5
                                    and build_j >= 0 and build_j < 5
                                    and (build_i != move_i or build_j != move_j) # check if worker is not building on its own location
                                    and self.workers[build_i][build_j] == 0
                                    and abs(self.levels[build_i][build_j]
                                          - self.levels[move_i][move_j]) <= 1 # check if worker can build on that level
                                    and self.levels[build_i][build_j] < 4):

                                    actions.append([cur_i, cur_j, move_i, move_j, build_i, build_j])
        
        return actions

    def move(self, cur_i, cur_j, move_i, move_j, build_i, build_j) -> None:
        """
        Move the worker to the new location and build a building.
        :param cur_i: The row of the worker.
        :param cur_j: The column of the worker.
        :param move_i: The row of the new location.
        :param move_j: The column of the new location.
        :param build_i: The row of the building.
        :param build_j: The column of the building.
        """
        worker = self.workers[cur_i][cur_j]
        self.workers[cur_i][cur_j] = 0
        self.workers[move_i][move_j] = worker
        self.levels[build_i][build_j] += 1
        self.board[0] = self.workers
        self.board[1] = self.levels
        player = 1 if worker > 0 else 2
        self.check_win(player)

    def check_win(self, player) -> None:
        """
        Check if the current player has won the game.
        :param player: The player whose turn it is.
        """
        player_workers = [i * (-1 ** (player - 1)) for i in [1, 2]]
        for worker in player_workers:
            worker_pos = np.argwhere(self.workers == worker)
            for pos in worker_pos:
                if self.levels[pos[0]][pos[1]] == 3:
                    self.set_winner(player)
                    return player
        return 0

    def set_winner(self, player) -> None:
        """
        Set the winner of the game.
        :param player: The player who has won the game.
        """
        self.winner = player

    def reset(self) -> None:
        """
        Reset the board to its initial state.
        """
        self.board = np.zeros((2, 5, 5))
        self.workers = np.zeros((5, 5))
        self.levels = np.zeros((5, 5))
        self.winner = 0
        self.init_board()

    def get_reward(self, player) -> int:
        """
        Get the reward for the player.
        :param player: The player whose turn it is.
        :return: The reward for the player.
        """
        if self.winner == player:
            return 1
        elif self.winner == -player:
            return -1
        else:
            return 0
