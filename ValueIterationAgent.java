package ticTacToe;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A Value Iteration Agent, only very partially implemented. The methods to
 * implement are: (1) {@link ValueIterationAgent#iterate} (2)
 * {@link ValueIterationAgent#extractPolicy}
 * 
 * You may also want/need to edit {@link ValueIterationAgent#train} - feel free
 * to do this, but you probably won't need to.
 * 
 * @author ae187
 *
 */
public class ValueIterationAgent extends Agent {

	/**
	 * This map is used to store the values of states
	 */
	Map<Game, Double> valueFunction = new HashMap<Game, Double>();

	/**
	 * the discount factor
	 */
	double discount = 0.9;

	/**
	 * the MDP model
	 */
	TTTMDP mdp = new TTTMDP();

	/**
	 * the number of iterations to perform - feel free to change this/try out
	 * different numbers of iterations
	 */
	int k = 10;

	/**
	 * This constructor trains the agent offline first and sets its policy
	 */
	public ValueIterationAgent() {
		super();
		mdp = new TTTMDP();
		this.discount = 0.9;
		initValues();
		train();
	}

	/**
	 * Use this constructor to initialise your agent with an existing policy
	 * 
	 * @param p
	 */
	public ValueIterationAgent(Policy p) {
		super(p);

	}

	public ValueIterationAgent(double discountFactor) {

		this.discount = discountFactor;
		mdp = new TTTMDP();
		initValues();
		train();
	}

	/**
	 * Initialises the {@link ValueIterationAgent#valueFunction} map, and sets the
	 * initial value of all states to 0 (V0 from the lectures). Uses
	 * {@link Game#inverseHash} and {@link Game#generateAllValidGames(char)} to do
	 * this.
	 * 
	 */
	public void initValues() {

		List<Game> allGames = Game.generateAllValidGames('X');// all valid games where it is X's turn, or it's terminal.
		for (Game g : allGames)
			this.valueFunction.put(g, 0.0);

	}

	public ValueIterationAgent(double discountFactor, double winReward, double loseReward, double livingReward,
			double drawReward) {
		this.discount = discountFactor;
		mdp = new TTTMDP(winReward, loseReward, livingReward, drawReward);
	}

	/**
	 * 
	 * 
	 * /* Performs {@link #k} value iteration steps. After running this method, the
	 * {@link ValueIterationAgent#valueFunction} map should contain the (current)
	 * values of each reachable state. You should use the {@link TTTMDP} provided to
	 * do this.
	 * 
	 *
	 */
	public void iterate() {
		Map<Game, Double> iteratedVals = new HashMap<>(); // creating a Hashmap to store each game state's value

		for (int i = 0; i < k; i++) { // for all k iterations get the game states
			for (Game g : valueFunction.keySet()) {
				double maxVal = Double.NEGATIVE_INFINITY; // initialize maximum move value to negative infinity

				if (!g.isTerminal()) { // get all possible moves for the game state if the game state is not
											// terminal
					for (Move m : g.getPossibleMoves()) {
						double moveVal = 0.0; // initialize move value to 0

						List<TransitionProb> transitions = mdp.generateTransitions(g, m); // generate transitions and calculate bellman equations
						for (TransitionProb transition : transitions) {
							double transitionVal = transition.prob * (transition.outcome.localReward
									+ discount * valueFunction.get(transition.outcome.sPrime));
																								
																			
							moveVal += transitionVal; 
						}

						if (moveVal > maxVal) { // updating the move value if its the greatest value
							maxVal = moveVal;
						}
					}
				}

				if (!g.isTerminal()) { // updating the value to the hashmap
					iteratedVals.put(g, maxVal);
				} else {
					iteratedVals.put(g, 0.0);
				}

			}

			valueFunction.putAll(iteratedVals);
		}
	}
	/*
	 * YOUR CODE HERE
	 */

	/**
	 * This method should be run AFTER the train method to extract a policy
	 * according to {@link ValueIterationAgent#valueFunction} You will need to do a
	 * single step of expectimax from each game (state) key in
	 * {@link ValueIterationAgent#valueFunction} to extract a policy.
	 * 
	 * @return the policy according to {@link ValueIterationAgent#valueFunction}
	 */
	public Policy extractPolicy() {
		Policy policies = new Policy(); //create an object for the policy called policies to extract the policies

		for (Game g : valueFunction.keySet()) { //iterate through game states
			if (!g.isTerminal()) {
				double maxVal = Double.NEGATIVE_INFINITY; //initialize max value to negative infinity
				Move bestMove = null; //initialize the best move to null

				for (Move m : g.getPossibleMoves()) { // iterate through all moves for the state
					double moveVal = 0.0; // initialize the value for the current move

					List<TransitionProb> transitions = mdp.generateTransitions(g, m);
					for (TransitionProb transition : transitions) {
						double transitionVal = transition.prob * (transition.outcome.localReward
								+ discount * valueFunction.get(transition.outcome.sPrime));
						moveVal += transitionVal;
					}

					if (moveVal > maxVal) {
						maxVal = moveVal;
						bestMove = m;
					}
				}

				policies.policy.put(g, bestMove); // Add best move for the current state to the policy
			}
		}

		return policies;
	}

	/*
	 * YOUR CODE HERE
	 */

	/**
	 * This method solves the mdp using your implementation of
	 * {@link ValueIterationAgent#extractPolicy} and
	 * {@link ValueIterationAgent#iterate}.
	 */
	public void train() {
		/**
		 * First run value iteration
		 */
		this.iterate();
		/**
		 * now extract policy from the values in
		 * {@link ValueIterationAgent#valueFunction} and set the agent's policy
		 * 
		 */

		super.policy = extractPolicy();

		if (this.policy == null) {
			System.out.println("Unimplemented methods! First implement the iterate() & extractPolicy() methods");
			// System.exit(1);
		}

	}

	public static void main(String a[]) throws IllegalMoveException {
		// Test method to play the agent against a human agent.
		ValueIterationAgent agent = new ValueIterationAgent();
		HumanAgent d = new HumanAgent();

		Game g = new Game(agent, d, d);
		g.playOut();

	}
}
