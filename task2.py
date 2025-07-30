# -- START OF YOUR CODERUNNER SUBMISSION CODE
# INCLUDE ALL YOUR IMPORTS HERE
import numpy as np
# The task2 solution calculates probabilities related toa  joint distribution of 
# of two random variables which is X and Y, these represent the time required for two phases
# of a security scan process which is finding and fixing. 
# It also calculates the probability of Y=8 given that a case us tested poisitive, using Bayes' Theorem.
# The function takes inputs: the total number of cases (num, a joint distribution table, and conditional probailities).
# 
# Key steps: 
# 
# 1. The joint distribution table is converted 
#    into probabilities by dividing each entry by the total number of cases (num)
#    This will give the join probability distribution P( X = x, Y = y).
#
# 2. The marginal distributions of X and Y are calculated by summing the joint probabilities 
#    over the other variable.
#    - P(X = x) = sum over y of P(X = x, Y = y).
#    - P(Y = y) = sum over x of P(X = x, Y = y).
#
# 3. The probability that X is between 3 and 4 is calculated
#     by summing the probabilities of X = 3 and X = 4 from the marginal distribution of X.
#    - prob1 = P(X = 3) + P(X = 4).
#
# 4. The probability that X + Y ≤ 10 is calculated by iterating over all combinations of X and Y and summing the joint probabilities where X + Y ≤ 10:
#    - prob2 = sum over x, y where x + y ≤ 10 of P(X = x, Y = y).
# 
# 5. The probability that Y = 8 given that a case is tested positive (T) is calculated using Bayes' ther
#

# The function returns the following probabilities as a tuple:
# 1. prob1: The probability that X is between 3 and 4.
# 2. prob2: The probability that X + Y ≤ 10.
# 3. prob3: The probability that Y = 8 given that a case is tested positive.

def Task2(num, table, probs):

    PX2, PX3, PX4, PX5, PY6, PY7 = probs
    
    joint_table = np.array(table)
    joint_probabilities = joint_table / num
    
    x_distribution = np.sum(joint_probabilities, axis=0)  
    y_distribution = np.sum(joint_probabilities, axis=1)  
    
    prob1 = x_distribution[1] + x_distribution[2]
    
    prob2 = 0
    for y_idx in range(3): 
        y_value = y_idx + 6
        for x_idx in range(4):  
            x_value = x_idx + 2
            if x_value + y_value <= 10:
                prob2 += joint_probabilities[y_idx, x_idx]
    
    
    probability_of_T = 0
    probability_of_T += PX2 * x_distribution[0]
    probability_of_T += PX3 * x_distribution[1]
    probability_of_T += PX4 * x_distribution[2]
    probability_of_T += PX5 * x_distribution[3]
    
    contribution_from_y6_y7 = 0
    contribution_from_y6_y7 += PY6 * y_distribution[0]  
    contribution_from_y6_y7 += PY7 * y_distribution[1]  
    

    if probability_of_T > 0:
        prob3 = (probability_of_T - contribution_from_y6_y7) / probability_of_T
    else:
        prob3 = 0
    
    return (prob1, prob2, prob3)
# -- END OF YOUR CODERUNNER SUBMISSION CODE