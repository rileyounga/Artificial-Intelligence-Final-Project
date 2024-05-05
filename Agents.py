import numpy as np
import torch
from torch import nn

ACTION_SPACE = 25 * 25 * 25

class RandomAgent():
    """
    This class will define a random agent that will play the game by selecting random legal moves.
    """
    
    def __init__(self, player) -> None:
        self.player = player
        self.type = "Random"

    def select_move(self, board) -> tuple:
        """
        Select a random legal move.
        :param board: The current state of the board.
        :return: A tuple containing the move and the build.
        """
        legal_actions = board.get_legal_moves(self.player)
        # since random.choice will throw an error if the list is empty,
        # we perform a win check here rather than in Board
        if len(legal_actions) == 0:
            opponent = 1 if self.player == 2 else 2
            board.set_winner(opponent)
            return None
        return legal_actions[np.random.choice(len(legal_actions))]
    
    def get_type(self) -> str:
        """
        Return the type of the agent.
        """
        return self.type

class HeuristicAgent():
    """
    This class will define a heuristic agent that will play the game by selecting the best legal move.

    The heuristic is based on the following:
    - If the agent can win the game, it will.
    - If the agent can prevent the opponent from winning the game, it will.
    """

    def __init__(self, player) -> None:
        self.player = player
        self.type = "Heuristic"

    def select_move(self, board) -> tuple:
        """
        Select the best legal move based on the heuristic.
        :param board: The current state of the board.
        :return: A tuple containing the move and the build.
        
        """
        legal_actions = board.get_legal_moves(self.player)
        if len(legal_actions) == 0:
            opponent = 1 if self.player == 2 else 2
            board.set_winner(opponent)
            return None
        
        # Check if the agent can win the game
        for action in legal_actions:
            move_i, move_j, build_i, build_j = action[2:6]
            if board.levels[move_i][move_j] == 3:
                return action
            
        # Check if the opponentent can win the game
        opponent = 1 if self.player == 2 else 1
        best_opponent_action = None
        for opponent_action in board.get_legal_moves(opponent):
            move_i, move_j, build_i, build_j = opponent_action[2:6]
            if board.levels[move_i][move_j] == 3:
                best_opponent_action = opponent_action
                break

        # Check if the agent can prevent the opponent from winning the game
        if best_opponent_action is not None:
            for action in legal_actions:
                build_i, build_j = action[4:6]
                if build_i == best_opponent_action[2] and build_j == best_opponent_action[3]:
                    return action

        # Otherwise, randomly select a move
        return legal_actions[np.random.choice(len(legal_actions))]


    def get_type(self) -> str:
        """
        Return the type of the agent.
        """
        return self.type

class QAgent():
    """
    This class will define a Q-learning agent that will play the game by selecting the best legal move.
    """

    def __init__(self, player, gamma, epsilon, alpha) -> None:
        self.player = player
        self.type = "Q"
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.q_table = {}
        self.state = None
        self.action = None

    def select_move(self, board) -> tuple:
        """
        Select the best legal move based on the Q-learning algorithm.
        :param board: The current state of the board.
        :return: A tuple containing the move and the build.
        """
        legal_actions = board.get_legal_moves(self.player)
        if len(legal_actions) == 0:
            opponent = 1 if self.player == 2 else 2
            board.set_winner(opponent)
            return None
        
        state = str(board.get_state())
        if state not in self.q_table:
            self.q_table[state] = np.zeros(ACTION_SPACE)
        
        if np.random.uniform() < self.epsilon:
            self.action = legal_actions[np.random.choice(len(legal_actions))]
            # perform epsilon decay
            self.epsilon *= 0.999
        else:
            self.action = legal_actions[np.argmax(self.q_table[state])]
        
        return self.action
    
    def update_q_table(self, board, reward) -> None:
        """
        Update the Q-table based on the reward.
        :param board: The current state of the board.
        :param reward: The reward for the player.
        """
        state = str(board.get_state())
        next_state = str(board.get_state())
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(ACTION_SPACE)
        
        self.q_table[state][self.action] += self.alpha * \
            (reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state][self.action])

    def get_type(self) -> str:
        """
        Return the type of the agent.
        """
        return self.type


class DQNAgent():
    """
    This class will define the DQN agent.
    The DQN agent will use a neural network to make decisions based on the current state of the board.
    """

    def __init__(self, player, gamma, epsilon, alpha) -> None:
        self.player = player
        self.q_network = nn.Sequential(
            nn.Linear(50, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, ACTION_SPACE + 1)
        )
        self.target_network = nn.Sequential(
            nn.Linear(50, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, ACTION_SPACE + 1)
        )
        self.q_network_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=alpha * 0.01)
        self.gamma = gamma
        self.epsilon = epsilon
        self.memory = []
        self.memory_size = 1000
        self.batch_size = 32
        self.action = None

    def select_move(self, board) -> tuple:
        """
        Select the best legal move based on the Q-learning algorithm.
        :param board: The current state of the board.
        :return: A tuple containing the move and the build.
        """
        legal_actions = board.get_legal_moves(self.player)
        max_action = len(legal_actions)
        if len(legal_actions) == 0:
            opponent = 1 if self.player == 2 else 2
            board.set_winner(opponent)
            return None
        
        state = torch.tensor(board.get_state().flatten(), dtype=torch.float32).unsqueeze(0)

        if np.random.uniform() < self.epsilon:
            self.action = legal_actions[np.random.choice(len(legal_actions))]
            self.epsilon *= 0.999
        else:
            q_values = self.q_network(state)
            self.action = legal_actions[torch.argmax(q_values[0][:max_action]).item()]
        
        return self.action
    
    def update_q_network(self, board, reward) -> None:
        """
        Update the Q-network based on the reward.
        :param board: The current state of the board.
        :param reward: The reward for the player.
        """
        self.memory.append((board.get_state().flatten(), self.action, reward))
        # remove the oldest memory if the memory size is exceeded
        if len(self.memory) > self.memory_size:
            del self.memory[0]
        
        if len(self.memory) < self.batch_size:
            return
        
        # sample a batch from the memory
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states, actions, rewards = zip(*[self.memory[i] for i in batch])
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)

        # calculate the target Q-values
        q_values = self.q_network(states)
        next_q_values = self.target_network(states)
        next_q_values = torch.max(next_q_values, dim=1)[0]
        target_q_values = rewards + self.gamma * next_q_values

        # decode actions to indices of the Q-values
        action_indices = [action[0]*25*25 + action[1]*25 + action[2] for action in actions]
        q_values[range(self.batch_size), action_indices] = target_q_values

        # update the Q-network
        loss = nn.MSELoss()(q_values, self.q_network(states))
        self.q_network_optimizer.zero_grad()
        loss.backward()
        self.q_network_optimizer.step()

    def get_type(self) -> str:
        """
        Return the type of the agent.
        """
        return "DQN"
        