"""
Asset selling driver script
"""

from collections import namedtuple
import pandas as pd
import numpy as np
from AssetSellingModel import AssetSellingModel
from AssetSellingPolicy import AssetSellingPolicy
import matplotlib.pyplot as plt
from copy import copy
import math

if __name__ == "__main__":
    # read in policy parameters from an Excel spreadsheet, "asset_selling_policy_parameters.xlsx"
    sheet1 = pd.read_excel("asset_selling_policy_parameters.xlsx", sheet_name="Sheet1")
    params = zip(sheet1['param1'], sheet1['param2'])
    param_list = list(params)
    sheet2 = pd.read_excel("asset_selling_policy_parameters.xlsx", sheet_name="Sheet2")
    sheet3 = pd.read_excel("asset_selling_policy_parameters.xlsx", sheet_name="Sheet3")

    # This dataframe contains the probabilities of prices trending up, sideways, or down
    biasdf = pd.read_excel("asset_selling_policy_parameters.xlsx", sheet_name="Sheet4", index_col=0)

    # Get the name of the policy that we are trialling in this model run
    policy_selected = sheet3['Policy'][0]

    # Read in the projection term, the initial price, and initial price change bias
    T = sheet3['TimeHorizon'][0]
    initPrice = sheet3['InitialPrice'][0]
    initBias = sheet3['InitialBias'][0]

    # Create dictionary of parameters for the exogenous info model. These are:
    #   - The mean change of an up-step
    #   - The mean change of an down-step
    #   - The variance of the price process
    #   - The dataframe of probabilities determining whether the market steps up or down.
    exog_params = {'UpStep':sheet3['UpStep'][0],'DownStep':sheet3['DownStep'][0],'Variance':sheet3['Variance'][0],'biasdf':biasdf}

    # Read in meta-parameters, including the number of itterations and the steps at which to print progress to console
    nIterations = sheet3['Iterations'][0] 
    printStep = sheet3['PrintStep'][0]
    printIterations = [0]
    printIterations.extend(list(reversed(range(nIterations-1,0,-printStep))))  

    print("exog_params ",exog_params)
   
    # create list containing the names of the policies to trial
    policy_names = ['sell_low', 'high_low', 'track']

    # create list of containing the names of the properties that define the model state and a dictionary of their
    # initial values
    state_names = ['price', 'resource','bias']
    init_state = {'price': initPrice, 'resource': 1,'bias':initBias}

    # create a list defining the possible values of the decisions that may be made at each step
    decision_names = ['sell', 'hold']

    # create an instance of the asset selling model by specifying:
    #   - the property names of the model's state and their initial values
    #   - the possible outcomes of the decision variable
    #   - the parameters needed to model the exogenous information process
    #   - the term of the projections
    M = AssetSellingModel(state_names, decision_names, init_state, exog_params, T)

    # create instance of the asset selling policy class using:
    #   - the instance of out asset selling model that we created above
    #   - a list containing the names of the policies we are trialling
    P = AssetSellingPolicy(M, policy_names)

    # set the initial time-step to t=0
    t = 0

    # set the previous price to the initial price
    prev_price = init_state['price']

    # make a policy_info dict object containing the parameter values required by each of our policies.
    policy_info = {'sell_low': param_list[0],
                   'high_low': param_list[1],
                   'track': param_list[2] + (prev_price,)}

    # The following code is used for all policies except the 'full_grid' policy
    if (not policy_selected =='full_grid'):
        print("Selected policy {}, time horizon {}, initial price {} and number of iterations {}".format(policy_selected,T,initPrice,nIterations))

        # Run the policy n times using and get the value of the objective function for each run
        contribution_iterations=[P.run_policy(param_list, policy_info, policy_selected, t) for ite in list(range(nIterations))]
        contribution_iterations = pd.Series(contribution_iterations)
        print("Contribution per iteration: ")
        print(contribution_iterations)

        # Calculate mean contribution
        cum_avg_contrib = contribution_iterations.expanding().mean()
        print("Cumulative average contribution per iteration: ")
        print(cum_avg_contrib)
        
        # Plot the results
        fig, axsubs = plt.subplots(1,2,sharex=True,sharey=True)
        fig.suptitle("Asset selling using policy {} with parameters {} and T {}".format(policy_selected,policy_info[policy_selected],T) )
        i = np.arange(0, nIterations, 1)
        
        axsubs[0].plot(i, cum_avg_contrib, 'g')
        axsubs[0].set_title('Cumulative average contribution')
          
        axsubs[1].plot(i, contribution_iterations, 'g')
        axsubs[1].set_title('Contribution per iteration')
        
    
        # Create a big subplot
        ax = fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axes
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

        ax.set_ylabel('USD', labelpad=0) # Use argument `labelpad` to move label downwards.
        ax.set_xlabel('Iterations', labelpad=10)
        
        plt.show()

    # The following code is used for the 'full_grid' policy
    else:
        # obtain the theta values to carry out a full grid search
        grid_search_theta_values = P.grid_search_theta_values(sheet2['low_min'], sheet2['low_max'], sheet2['high_min'], sheet2['high_max'], sheet2['increment_size'])
        # use those theta values to calculate corresponding contribution values
        
        contribution_iterations = [P.vary_theta(param_list, policy_info, "high_low", t, grid_search_theta_values[0]) for ite in list(range(nIterations))]
        
        contribution_iterations_arr = np.array(contribution_iterations)
        cum_sum_contrib = contribution_iterations_arr.cumsum(axis=0)
        nElem = np.arange(1,cum_sum_contrib.shape[0]+1).reshape((cum_sum_contrib.shape[0],1))
        cum_avg_contrib=cum_sum_contrib/nElem
        print("cum_avg_contrib")
        print(cum_avg_contrib)
    
        # plot those contribution values on a heat map
        P.plot_heat_map_many(cum_avg_contrib, grid_search_theta_values[1], grid_search_theta_values[2], printIterations)
        
        