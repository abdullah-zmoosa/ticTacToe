package ticTacToe;

import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.Set;

/**
 * A policy iteration agent. You should implement the following methods: (1)
 * {@link PolicyIterationAgent#evaluatePolicy}: this is the policy evaluation
 * step from your lectures (2) {@link PolicyIterationAgent#improvePolicy}: this
 * is the policy improvement step from your lectures (3)
 * {@link PolicyIterationAgent#train}: this is a method that should
 * runs/alternate (1) and (2) until convergence.
 * 
 * NOTE: there are two types of convergence involved in Policy Iteration:
 * Convergence of the Values of the current policy, and Convergence of the
 * current policy to the optimal policy. The former happens when the values of
 * the current policy no longer improve by much (i.e. the maximum improvement is
 * less than some small delta). The latter happens when the policy improvement
 * step no longer updates the policy, i.e. the current policy is already
 * optimal. The algorithm should stop when this happens.
 * 
 * @author ae187
 *
 */
public class PolicyIterationAgent extends Agent {

	/**
	 * This map is used to store the values of states according to the current
	 * policy (policy evaluation).
	 */
	HashMap<Game, Double> policyValues = new HashMap<Game, Double>();

	/**
	 * This stores the current policy as a map from {@link Game}s to {@link Move}.
	 */
	HashMap<Game, Move> curPolicy = new HashMap<Game, Move>();

	double discount = 0.9;

	/**
	 * The mdp model used, see {@link TTTMDP}
	 */
	TTTMDP mdp;

	/**
	 * loads the policy from file if one exists. Policies should be stored in .pol
	 * files directly under the project folder.
	 */
	public PolicyIterationAgent() {
		super();
		this.mdp = new TTTMDP();
		initValues();
		initRandomPolicy();
		train();

	}

	/**
	 * Use this constructor to initialise your agent with an existing policy
	 * 
	 * @param p
	 */
	public PolicyIterationAgent(Policy p) {
		super(p);

	}

	/**
	 * Use this constructor to initialise a learning agent with default MDP
	 * paramters (rewards, transitions, etc) as specified in {@link TTTMDP}
	 * 
	 * @param discountFactor
	 */
	public PolicyIterationAgent(double discountFactor) {

		this.discount = discountFactor;
		this.mdp = new TTTMDP();
		initValues();
		initRandomPolicy();
		train();
	}

	/**
	 * Use this constructor to set the various parameters of the Tic-Tac-Toe MDP
	 * 
	 * @param discountFactor
	 * @param winningReward
	 * @param losingReward
	 * @param livingReward
	 * @param drawReward
	 */
	public PolicyIterationAgent(double discountFactor, double winningReward, double losingReward, double livingReward,
			double drawReward) {
		this.discount = discountFactor;
		this.mdp = new TTTMDP(winningReward, losingReward, livingReward, drawReward);
		initValues();
		initRandomPolicy();
		train();
	}

	/**
	 * Initialises the {@link #policyValues} map, and sets the initial value of all
	 * states to 0 (V0 under some policy pi ({@link #curPolicy} from the lectures).
	 * Uses {@link Game#inverseHash} and {@link Game#generateAllValidGames(char)} to
	 * do this.
	 * 
	 */
	public void initValues() {
		List<Game> allGames = Game.generateAllValidGames('X');// all valid games where it is X's turn, or it's terminal.
		for (Game g : allGames)
			this.policyValues.put(g, 0.0);

	}

	/**
	 * You should implement this method to initially generate a random policy, i.e.
	 * fill the {@link #curPolicy} for every state. Take care that the moves you
	 * choose for each state ARE VALID. You can use the
	 * {@link Game#getPossibleMoves()} method to get a list of valid moves and
	 * choose randomly between them.
	 */
	public void initRandomPolicy() {
		Random randomPolicy = new Random(); // create an object to generate random moves

		for (Game g : policyValues.keySet()) {
			if (!g.isTerminal()) {
				List<Move> possiblemoves = g.getPossibleMoves();
				if (!possiblemoves.isEmpty()) {
					Move randomMove = possiblemoves.get(randomPolicy.nextInt(possiblemoves.size())); // Select a random
																										// move from the
																										// valid moves
					curPolicy.put(g, randomMove); // choose a move randomly to add to the policy
				}
			}
		}
		/*
		 * YOUR CODE HERE
		 */
	}

	/**
	 * Performs policy evaluation steps until the maximum change in values is less
	 * than {@code delta}, in other words until the values under the currrent policy
	 * converge. After running this method, the
	 * {@link PolicyIterationAgent#policyValues} map should contain the values of
	 * each reachable state under the current policy. You should use the
	 * {@link TTTMDP} {@link PolicyIterationAgent#mdp} provided to do this.
	 *
	 * @param delta
	 */
	protected void evaluatePolicy(double delta) {
		boolean isConverged;
		do { // looping until delta reaches threshold
			isConverged = true;

			for (Game g : curPolicy.keySet()) {
				double oldVal = policyValues.get(g); // Store the current value of the state
				double newVal = 0.0; // initialize the new value for the state

				for (Move m : g.getPossibleMoves()) {
					for (TransitionProb transition : mdp.generateTransitions(g, m)) {
						double transitionValue = transition.prob * (transition.outcome.localReward
								+ (discount * policyValues.get(transition.outcome.sPrime)));
						newVal += transitionValue;
					}
				}

				policyValues.put(g, newVal); // update the value of the state in the policy values
				if (Math.abs(oldVal - newVal) >= delta) { // checking if delta threshold is reached and ending loop
					isConverged = false;
				}
			}
		} while (!isConverged);

		/* YOUR CODE HERE */

	}

	/**
	 * This method should be run AFTER the
	 * {@link PolicyIterationAgent#evaluatePolicy} train method to improve the
	 * current policy according to {@link PolicyIterationAgent#policyValues}. You
	 * will need to do a single step of expectimax from each game (state) key in
	 * {@link PolicyIterationAgent#curPolicy} to look for a move/action that
	 * potentially improves the current policy.
	 * 
	 * @return true if the policy improved. Returns false if there was no
	 *         improvement, i.e. the policy already returned the optimal actions.
	 */
	protected boolean improvePolicy() {
		Set<Game> states = curPolicy.keySet();
		boolean improvePolicy = false;

		for (Game g : states) {
			if (!g.isTerminal()) {
				Move currentMove = curPolicy.get(g); // assign current move to the state in the current policy
				Move bestMove = null;
				double maxQVal = Double.NEGATIVE_INFINITY;

				for (Move m : g.getPossibleMoves()) { // get all possible moves in state and find Q value using bellman
														// equation
					double QVal = 0.0;

					for (TransitionProb transition : mdp.generateTransitions(g, m)) {
						double transitionValue = transition.prob * (transition.outcome.localReward
								+ (discount * policyValues.get(transition.outcome.sPrime)));
						QVal += transitionValue;
					}

					if (QVal > maxQVal) { // find maximum Q value and assign it as the best move
						maxQVal = QVal;

						bestMove = m;
					}
				}

				if (!currentMove.equals(bestMove)) { // check the current move and the best move
					curPolicy.put(g, bestMove); // Update the policy with the best move
					improvePolicy = true;
				}
			}
		}

		return improvePolicy;
		/* YOUR CODE HERE */
	}

	/**
	 * The (convergence) delta
	 */
	double delta = 0.1;

	/**
	 * This method should perform policy evaluation and policy improvement steps
	 * until convergence (i.e. until the policy no longer changes), and so uses your
	 * {@link PolicyIterationAgent#evaluatePolicy} and
	 * {@link PolicyIterationAgent#improvePolicy} methods.
	 */
	public void train() {
		boolean policyChanged; // check if the policy has changed after policy evaluation and policy
								// improvement

		do {
			evaluatePolicy(delta); // do policy evaluation and update values of states

			policyChanged = improvePolicy(); // do policy improvement to update the policy

		} while (policyChanged);

		super.policy = new Policy(curPolicy); // Set the policy of agent to the final policy

	}

	public static void main(String[] args) throws IllegalMoveException {
		/**
		 * Test code to run the Policy Iteration Agent agains a Human Agent.
		 */
		PolicyIterationAgent pi = new PolicyIterationAgent();

		HumanAgent h = new HumanAgent();

		Game g = new Game(pi, h, h);

		g.playOut();

	}

}
