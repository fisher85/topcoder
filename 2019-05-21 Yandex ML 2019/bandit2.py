class Environment:
	    def __init__(self, variants, payouts, n_trials, variance=False):
	        self.variants = variants
	        if variance:
	            self.payouts = np.clip(payouts + np.random.normal(0, 0.04, size=len(variants)), 0, .2)
	        else:
	            self.payouts = payouts
	        #self.payouts[5] = self.payouts[5] if i < n_trials/2 else 0.1
	        self.n_trials = n_trials
	        self.total_reward = 0
	        self.n_k = len(variants)
	        self.shape = (self.n_k, n_trials)
	        
	    def run(self, agent):
	        """Run the simulation with the agent. 
	        agent must be a class with choose_k and update methods."""
	        
	        for i in range(self.n_trials):
	            # agent makes a choice
	            x_chosen = agent.choose_k()
	            # Environment returns reward
	            reward = np.random.binomial(1, p=self.payouts[x_chosen])
	            # agent learns of reward
	            agent.reward = reward
	            # agent updates parameters based on the data
	            agent.update()
	            self.total_reward += reward
	        
	        agent.collect_data()
	        
	        return self.total_reward