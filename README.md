# Risk-Averse Distributional Reinforcement Learning

This package contains code for CVaR Value Iteration, CVaR Q-learning and Deep CVaR Q-learning. 


## Installation

Using GPU during training is highly recommended (but not required)

    pip3 install tensorflow-gpu

Next install baselines located in `software/baselines`. Navigate to the location and run
    
    pip3 install -e .

Next install PLE located in `software/PyGame-Learning-Environment`. Navigate to the location and run
    
    pip3 install -e .

Lastly, install the cvar package located in `code`. Navigate to the location and run

    pip3 install -e .

### CVaR Value Iteration, CVaR Q-learning

See readme in `cvar/gridworld`


### Deep CVaR Q-learning 

See readme in `cvar/dqn/scripts`
