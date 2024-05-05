from Board import *
from Agents import *
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors


verbose = False

def main():
    agent_types = [RandomAgent, HeuristicAgent, QAgent, DQNAgent]
    heatmap = np.zeros((4, 4))
    num_games = 100
    batches = 10
    rerun = False

    if rerun:
        for _ in range(int(num_games/batches)):
            print("\033[H\033[J") # clear the screen
            print("Progress:", "=" * int(20 * _ / (num_games/batches)) + ">" + "-" * int(20 * (1 - _ / (num_games/batches)) - 1))
            matchups = []
            for agent_type1 in agent_types:
                for agent_type2 in agent_types:
                    # Skip duplicate matchups
                    if agent_type1 == agent_type2:
                        continue
                    if (agent_type1, agent_type2) in matchups or (agent_type2, agent_type1) in matchups:
                        continue
                    else:
                        matchups.append((agent_type1, agent_type2))
                    
                    if agent_type1 in [QAgent, DQNAgent]:
                        # agents are initialized with learning rate, discount factor, and epsilon
                        # future tests could hyperparameter tune these values
                        agent1 = agent_type1(1, 0.1, 0.9, 0.1)
                    else:
                        agent1 = agent_type1(1)
                    if agent_type2 in [QAgent, DQNAgent]:
                        agent2 = agent_type2(2, 0.1, 0.9, 0.1)
                    else:
                        agent2 = agent_type2(2)
                    
                    for _ in range(batches):
                        winner = play_game(agent1, agent2)
                        if winner == 1:
                            heatmap[agent_types.index(agent_type1), agent_types.index(agent_type2)] += 1
                        elif winner == 2:
                            heatmap[agent_types.index(agent_type2), agent_types.index(agent_type1)] += 1
        
        # Save the heatmap to a file
        np.save("heatmap.npy", heatmap)
    else:
        heatmap = np.load("heatmap.npy")

    # plot heatmap of agents wins relative to each other
    fig, ax = plt.subplots()

    # the heatmap is going to heavily biased towards agents with 90-100 wins and 0-10 wins
    # so we need a diverging colormap to be less sensitive to the extremes
    norm = colors.LogNorm(vmin=heatmap.min() + 1, vmax=heatmap.max())

    cbar = ax.imshow(heatmap, cmap="coolwarm", norm=norm)

    agent_names = ["Random", "Heuristic", "Q", "DQN"]
    ax.set_xticks(np.arange(len(agent_names)))
    ax.set_yticks(np.arange(len(agent_names)))
    ax.set_xticklabels(agent_names)
    ax.set_yticklabels(agent_names)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(agent_names)):
        for j in range(len(agent_names)):
            ax.text(j, i, heatmap[i, j], ha="center", va="center", color="black")

    ax.set_title("Agent Wins")
    fig.tight_layout()

    plt.savefig("heatmap.png")

    # average across the rows to get the win rate of each agent
    # exclude the diagonal from the average since its zero
    heatmap_copy = np.copy(heatmap)
    np.fill_diagonal(heatmap_copy, np.nan)
    win_rates = np.nanmean(heatmap_copy, axis=1)
    for i in range(len(agent_names)):
        print(f"{agent_names[i]} win rate: {win_rates[i]}")


def play_game(agent1, agent2):
    """
    Play a game between two agents.
    :param agent1: The first agent.
    :param agent2: The second agent.
    :return: The winner of the game.
    """
    p1_type = agent1.get_type()
    p2_type = agent2.get_type()
    # Create a new board
    board = Board()
    if verbose:
        print(board.get_state())
        print()

    # Play the game
    while board.winner == 0:
        # Select a move for the first agent
        action = agent1.select_move(board)
        if action is None:
            break
        if verbose:
            print(f"Agent {agent1.get_type()} selected action: {action}")
        board.move(*action)
        if verbose:
            print(board.get_state())
            print()
        if p1_type == "Q":
            reward = board.get_reward(agent1.player)
            agent1.update_q_table(board, reward)
        elif p1_type == "DQN":
            reward = board.get_reward(agent1.player)
            agent1.update_q_network(board, reward)

        # Select a move for the second agent
        action = agent2.select_move(board)
        if action is None:
            break
        if verbose:
            print(f"Agent {agent2.get_type()} selected action: {action}")
        board.move(*action)
        if verbose:
            print(board.get_state())
            print()

        if p2_type == "Q":
            reward = board.get_reward(agent2.player)
            agent2.update_q_table(board, reward)
        elif p2_type == "DQN":
            reward = board.get_reward(agent2.player)
            agent2.update_q_network(board, reward)

    if p1_type == "Q":
        reward = board.get_reward(agent1.player)
        agent1.update_q_table(board, reward)
    elif p1_type == "DQN":
        reward = board.get_reward(agent1.player)
        agent1.update_q_network(board, reward)

    if p2_type == "Q":
        reward = board.get_reward(agent2.player)
        agent2.update_q_table(board, reward)
    elif p2_type == "DQN":
        reward = board.get_reward(agent2.player)
        agent2.update_q_network(board, reward)

    if verbose:
        print(f"Winner: {board.winner}")
        print()

    return board.winner


if __name__ == "__main__":
    main()
    