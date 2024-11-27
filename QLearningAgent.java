package ticTacToe;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.NoSuchElementException;
import java.util.Random;

/**
 * A Q-Learning agent with a Q-Table, i.e. a table of Q-Values. This table is
 * implemented in the {@link QTable} class.
 * 
 * The methods to implement are: (1) {@link QLearningAgent#train} (2)
 * {@link QLearningAgent#extractPolicy}
 * 
 * Your agent acts in a {@link TTTEnvironment} which provides the method
 * {@link TTTEnvironment#executeMove} which returns an {@link Outcome} object,
 * in other words an [s,a,r,s']: source state, action taken, reward received,
 * and the target state after the opponent has played their move. You may
 * want/need to edit {@link TTTEnvironment} - but you probably won't need to.
 * 
 * @author ae187
 */

public class QLearningAgent extends Agent {

	/**
	 * The learning rate, between 0 and 1.
	 */
	double alpha = 0.1;

	/**
	 * The number of episodes to train for
	 */
	int numEpisodes = 10000;

	/**
	 * The discount factor (gamma)
	 */
	double discount = 0.9;

	/**
	 * The epsilon in the epsilon greedy policy used during training.
	 */
	double epsilon = 0.1;

	/**
	 * This is the Q-Table. To get an value for an (s,a) pair, i.e. a (game, move)
	 * pair.
	 * 
	 */

	QTable qTable = new QTable();

	/**
	 * This is the Reinforcement Learning environment that this agent will interact
	 * with when it is training. By default, the opponent is the random agent which
	 * should make your q learning agent learn the same policy as your value
	 * iteration and policy iteration agents.
	 */
	TTTEnvironment env = new TTTEnvironment();

	/**
	 * Construct a Q-Learning agent that learns from interactions with
	 * {@code opponent}.
	 * 
	 * @param opponent     the opponent agent that this Q-Learning agent will
	 *                     interact with to learn.
	 * @param learningRate This is the rate at which the agent learns. Alpha from
	 *                     your lectures.
	 * @param numEpisodes  The number of episodes (games) to train for
	 */
	public QLearningAgent(Agent opponent, double learningRate, int numEpisodes, double discount) {
		env = new TTTEnvironment(opponent);
		this.alpha = learningRate;
		this.numEpisodes = numEpisodes;
		this.discount = discount;
		initQTable();
		train();
	}

	/**
	 * Initialises all valid q-values -- Q(g,m) -- to 0.
	 * 
	 */

	protected void initQTable() {
		List<Game> allGames = Game.generateAllValidGames('X');// all valid games where it is X's turn, or it's terminal.
		for (Game g : allGames) {
			List<Move> moves = g.getPossibleMoves();
			for (Move m : moves) {
				this.qTable.addQValue(g, m, 0.0);
				// System.out.println("initing q value. Game:"+g);
				// System.out.println("Move:"+m);
			}

		}

	}

	/**
	 * Uses default parameters for the opponent (a RandomAgent) and the learning
	 * rate (0.2). Use other constructor to set these manually.
	 */
	public QLearningAgent() {
		this(new RandomAgent(), 0.1, 30000, 0.9);

	}

	/**
	 * Implement this method. It should play {@code this.numEpisodes} episodes of
	 * Tic-Tac-Toe with the TTTEnvironment, updating q-values according to the
	 * Q-Learning algorithm as required. The agent should play according to an
	 * epsilon-greedy policy where with the probability {@code epsilon} the agent
	 * explores, and with probability {@code 1-epsilon}, it exploits.
	 * 
	 * At the end of this method you should always call the {@code extractPolicy()}
	 * method to extract the policy from the learned q-values. This is currently
	 * done for you on the last line of the method.
	 */

	public void train() {
		Random randomQ = new Random();

		for (int episodes = 0; episodes < numEpisodes; episodes++) { // Loop through the episodes
			env = new TTTEnvironment(); //create an envirnoment for episode
			Game g = env.getCurrentGameState(); //get game state in episode

			while (!g.isTerminal()) {
				List<Move> possibleMoves = env.getPossibleMoves(); //get all possible moves in game state

				if (!possibleMoves.isEmpty()) { 
					Move selectedMove;
					if (randomQ.nextDouble() > epsilon) { //if the Q value greater than epsilon select an action
					    Move bestMove = null;
					    double maxQVal = Double.NEGATIVE_INFINITY;

					    for (Move m : possibleMoves) {
					        double qVal = qTable.getQValue(g, m);
					        if (qVal > maxQVal) {
					            maxQVal = qVal;
					            bestMove = m;
					        }
					    }

					    if (bestMove != null) {
					        selectedMove = bestMove;
					    } else {
					        throw new NoSuchElementException(); //throw exception if best move not found
					    }
					} else { //explore other random moves
					    selectedMove = possibleMoves.get(randomQ.nextInt(possibleMoves.size()));
					}

					try { //execute the move and store it in an object
					    Outcome outcomeMove = env.executeMove(selectedMove);
					    List<Move> updatedMoves = env.getPossibleMoves(); //get possible moves after updating
					    double newQVal = 0.0;

					    if (!updatedMoves.isEmpty()) {
					        Move maxUpdatedMove = null;
					        double maxQVal = Double.NEGATIVE_INFINITY;

					        for (Move m : updatedMoves) {
					            double qVal = qTable.getQValue(outcomeMove.sPrime, m);
					            if (qVal > maxQVal) {
					                maxQVal = qVal;
					                maxUpdatedMove = m;
					            }
					        }

					        if (maxUpdatedMove != null) {
					            newQVal = qTable.getQValue(outcomeMove.sPrime, maxUpdatedMove);

					            if (!outcomeMove.sPrime.isTerminal()) { //find moves if new state is not terminal
					                for (Move m : updatedMoves) {
					                    double qVal = qTable.getQValue(outcomeMove.sPrime, m);
					                    if (qVal > newQVal) {
					                        newQVal = qVal;
					                    }
					                }
					            }
					        }
					    }

					    double currentQVal = qTable.getQValue(outcomeMove.s, selectedMove); //get current Q value and update it with the equation
					    double updatedQVal = (1 - alpha) * currentQVal + alpha * (outcomeMove.localReward + discount * newQVal); 

					    qTable.addQValue(outcomeMove.s, selectedMove, updatedQVal);
					} catch (IllegalMoveException e) {  //Handle illegal moves and print an error
					    e.printStackTrace();
					}
				}
			}
		}


		/*
		 * YOUR CODE HERE
		 */

		// --------------------------------------------------------
		// you shouldn't need to delete the following lines of code.
		this.policy = extractPolicy();
		if (this.policy == null) {
			System.out.println("Unimplemented methods! First implement the train() & extractPolicy methods");
			// System.exit(1);
		}
	}

	/**
	 * Implement this method. It should use the q-values in the {@code qTable} to
	 * extract a policy and return it.
	 *
	 * @return the policy currently inherent in the QTable
	 */
	public Policy extractPolicy() {
		Policy optimalPolicy = new Policy();

	    // Loop through the entries of the hashmap in the Q table
	    for (Entry<Game, HashMap<Move, Double>> entry : qTable.entrySet()) {
	        Game g = entry.getKey(); // Get game state
	        Map<Move, Double> moveMap = entry.getValue(); // Get best move and Q value

	        Move maxMove = null;
	        double maxQValue = Double.NEGATIVE_INFINITY;

	        // Iterate through the entries in map and fine maximum Q Value
	        for (Map.Entry<Move, Double> moveEntry : moveMap.entrySet()) {
	            Move currentMove = moveEntry.getKey();
	            double currentQValue = moveEntry.getValue();

	            if (currentQValue > maxQValue) {
	                maxQValue = currentQValue;
	                maxMove = currentMove;
	            }
	        }

	        // Add game state and best move with the maximum Q Value to the policy
	        optimalPolicy.policy.put(g, maxMove);
	    }

	    return optimalPolicy;
	}
		/*
		 * YOUR CODE HERE
		 */

	public static void main(String a[]) throws IllegalMoveException {
		// Test method to play your agent against a human agent (yourself).
		QLearningAgent agent = new QLearningAgent();

		HumanAgent d = new HumanAgent();

		Game g = new Game(agent, d, d);
		g.playOut();

	}

}
