"""
Asset selling model class
Adapted from code by Donghun Lee (c) 2018

"""
from collections import namedtuple
import numpy as np

class AssetSellingModel():
    """
    Base class for model
    """

    def __init__(self, state_variable, decision_variable, state_0, exog_0,T=10, exog_info_fn=None, transition_fn=None,
                 objective_fn=None, seed=20180529):
        """
        Initializes the model

        :param state_variable: list(str) - state variable dimension names
        :param decision_variable: list(str) - decision variable dimension names
        :param state_0: dict - needs to contain at least the information to populate initial state using state_names
        :param exog_info_fn: function - calculates relevant exogenous information
        :param transition_fn: function - takes in decision variables and exogenous information to describe how the state
               evolves
        :param objective_fn: function - calculates contribution at time t
        :param seed: int - seed for random number generator
        """

        # Create initial_args property, which contains the projection seed, the projection term, and the parameters
        # needed to model the exogenous information process
        self.initial_args = {'seed': seed,'T': T,'exog_params':exog_0}

        # Update the dataframe containing the probabilities of an up/down trend so that the probabilities are cumulative
        exog_params = self.initial_args['exog_params']
        biasdf = exog_params['biasdf']
        biasdf = biasdf.cumsum(axis=1)
        self.initial_args['exog_params'].update({'biasdf':biasdf})

        # Create instance of an RNG using specified seed
        self.prng = np.random.RandomState(seed)

        # Create initial state property
        self.initial_state = state_0

        # Create state variable property which lists the names of the state variable's properties
        self.state_variable = state_variable

        # Create decision variable property which lists the names of the decision variable's valid output values
        self.decision_variable = decision_variable

        # Create a named tuple containing the state variable's property values
        self.State = namedtuple('State', state_variable)
        self.state = self.build_state(state_0)

        # Instantiate a property for holding the possible decisions that may be made by our policies
        self.Decision = namedtuple('Decision', decision_variable)

        # TODO: Definition for this
        self.objective = 0.0


    def build_state(self, info):
        """
        this function gives a state containing all the state information needed

        :param info: dict - contains all state information
        :return: namedtuple - a state object
        """
        return self.State(*[info[k] for k in self.state_variable])

    def build_decision(self, info):
        """
        this function gives a decision

        :param info: dict - contains all decision info
        :return: namedtuple - a decision object
        """
        return self.Decision(*[info[k] for k in self.decision_variable])


    def exog_info_fn(self):
        """
        this function gives the exogenous information that is dependent on a random process (in the case of the the asset
        selling model, it is the change in price)

        :return: dict - updated price
        """
        # we assume that the change in price is normally distributed with mean bias and variance 2

        # Get args for exogenous information generation (mean under each scenario, the probability of each scenario
        # occurring, and the (constant) variance
        exog_params = self.initial_args['exog_params']
        
        # Transpose the dataframe containing the transition probabilities for the bias scenarios and extract the
        # probabilities for the current bias scenario
        biasdf = exog_params['biasdf'].T
        biasprob = biasdf[self.state.bias]
        
        # Simulate the bias state transition and get the mean to use for that bias state in the price transition
        # simulation
        coin = self.prng.random_sample()
        if (coin < biasprob['Up']):
            new_bias = 'Up'
            bias = exog_params['UpStep']
        elif (coin>=biasprob['Up'] and coin<biasprob['Neutral']):
            new_bias = 'Neutral'
            bias = 0
        else:
            new_bias = 'Down'
            bias = exog_params['DownStep']
         
        print("coin ", coin, " curr_bias ", self.state.bias, " new_bias ", new_bias)

        # Calculate the new price by incrementing the current price using a sample from a normal distribution with mean
        # equal to the mean change for the current bias state
        updated_price = self.state.price + self.prng.normal(bias, exog_params['Variance'])

        # we account for the fact that asset prices cannot be negative by setting the new price as 0 whenever the
        # random process gives us a negative price
        new_price = 0.0 if updated_price < 0.0 else updated_price

        # Return the new price and the new bias state
        return {"price": new_price, "bias":new_bias}

    def transition_fn(self, decision, exog_info):
        """
        this function takes in the decision and exogenous information to update the state

        :param decision: namedtuple - contains all decision info
        :param exog_info: any exogenous info (in this asset selling model,
               the exogenous info does not factor into the transition function)
        :return: dict - updated resource
        """

        # Return the number of assets held following each decision
        new_resource = 0 if decision.sell is 1 else self.state.resource
        return {"resource": new_resource}

    def objective_fn(self, decision, exog_info):
        """
        this function calculates the contribution, which depends on the decision and the price

        :param decision: namedtuple - contains all decision info
        :param exog_info: any exogenous info (in this asset selling model,
               the exogenous info does not factor into the objective function)
        :return: float - calculated contribution
        """

        # Parse the decision to a sale quantity
        sell_size = 1 if decision.sell is 1 and self.state.resource != 0 else 0

        # Get the decisions contribution to the objective function as the proceeds of the sale (or zero if no sale)
        obj_part = self.state.price * sell_size

        return obj_part

    def step(self, decision):
        """
        this function steps the process forward by one time increment by updating the sum of the contributions, the
        exogenous information and the state variable

        :param decision: namedtuple - contains all decision info
        :return: none
        """

        # Generate exogenous information (the new price and the new bias state)
        exog_info = self.exog_info_fn()

        # Update the value of the objective function based on the decision made and the new info received
        self.objective += self.objective_fn(decision, exog_info)

        # Update the properties of the state variable with the new price, new bias state, and new asset holding
        exog_info.update(self.transition_fn(decision, exog_info))
        self.state = self.build_state(exog_info)

