# -- START OF YOUR CODERUNNER SUBMISSION CODE
# INCLUDE ALL YOUR IMPORTS HERE
import numpy as np
import math
from dhCheck_Task1 import dhCheckCorrectness
# The solution below calculates cumalative distribution function (CDF) for triangular distribution. 
# It uses three parameters, the lower limit (a), upper limit (b) and the midpoint (c). 

# The solution handles four cases:
#   1. When x is simply below the lower limit(a), the probabilty will be given 0.
#   2. When x is between the lower limit(a) and the midpoint(c) it uses the formula: ((x-a)²)/((b-a)(c-a)).
#   3. When x is between the midpoint(c) and the upper limit(b) it uses the formula: 1-((b-x)²)/((b-a)(b-c)).
#   4. When x is simply above the upper limit the probability will be given 1. 
def triangular_cdf(x, a, b, c):
    if x <= a:
        return 0
    elif x <= c:
        return ((x - a)**2) / ((b - a) * (c - a))
    elif x <= b:
        return 1 - ((b - x)**2) / ((b - a) * (b - c))
    else:
        return 1

# The solution below calculates the mean of the triangular distribution, 
# finding the average of the three parameters: a, b and c.
# Formula : (a + b + c) / 3
def triangular_mean(a, b, c):
    return (a + b + c) / 3

# The solution below caculates the median of the triangular distribution. The median
# depends on whether the midpoint(c) is in the first(a) or second(b) half of the range.

# If c is in the upper limit, I use the formula a + sqrt((b-a)(c-a)/2).
# If c is in the lower limit, I use the formula b - sqrt((b-a)(b-c)/2).
# These formulas can help we find the value that divides the distribution in to equal halves. 
def triangular_median(a, b, c):

    if c >= (a + b) / 2:
        return a + math.sqrt((b - a) * (c - a) / 2)
    else:
        return b - math.sqrt((b - a) * (b - c) / 2)

# This function calculates the mean and variance for discrete probability distribution.
# The mean in this case is the weighted sum of all possible values, each value is weighted by its probability.
# The variance measures the spread of the distribution around the mean, which is then calculated as the weighted sum of
# of deviations squared from mean.
# This helps us determine the value and variability of annual occurences.
def calculate_discrete_mean_variance(numbers, probabilities):
    mean = sum(n * p for n, p in zip(numbers, probabilities))
    
    variance = sum(((n - mean) ** 2) * p for n, p in zip(numbers, probabilities))
    
    return mean, variance

# The function below helps geenerate random samples from lognormal distribution, with parameters mu and sigma.
# lognormal distribution is appropriate for modeling the impact flaws for flaw A, since security impacts, often follow
# a positive skewed distribution. 
# using lognormal function to efficiently generate samples.
def generate_lognormal_sample(mu, sigma, size):

    return np.random.lognormal(mu, sigma, size)

# This function creates random samples from a Pareto distribution using inverse transform method.
# The pareto distribution is ideal to model flaw B impact as it has "heavy tail" that represents rare security events.
# I use inverse CDF formula which is xm / (u^(1/alpha)) where "u" means uniformly distributed between 0 and 1.
# I have chosen this over the pareto function as convetnionally it gives us direct control over the distribution's characteristics and aligns with the
# standard approach in security risk.

def generate_pareto_sample(xm, alpha, size):

    u = np.random.uniform(0, 1, size) 
    
    return xm / (u ** (1/alpha))

# Task1 soltion integrates all the above functions to calculate the required prbabilities, mean, medians
# and ALE.
# It takes as input the parameters for the trianguler distribution, discrete probability distribution, 
# and Monte Carlo simulation then returns the required output. 
# 
# Function performs the steps in this order:
# 1. I calculate the probability that the asset value is not greater than a give point(point1)
# 2. I then compute the mean and median of the triangular distribution for the asset value.
# 3. I calculate the mean and the variance of the discrete probability distribution for annual occurrences.
# 4. Simulate the total impact of two flaws (A and B) using Monte Carlo methods.
# 5. Determine the probabiltiy that the total impact exceeds a given threshold (point2) and lies
#    within a specified range 
# 6. Compute the Single Loss Expectancy and Annualised Loss Expectancy based on the derived values.
def Task1(a, b, c, point1, number_set, prob_set, num, point2, mu, sigma, xm, alpha, point3, point4):
    prob1 = triangular_cdf(point1, a, b, c)
    
    MEAN_t = triangular_mean(a, b, c)
    MEDIAN_t = triangular_median(a, b, c)
    
    MEAN_d, VARIANCE_d = calculate_discrete_mean_variance(number_set, prob_set)
    
    impact_A = generate_lognormal_sample(mu, sigma, num)
    impact_B = generate_pareto_sample(xm, alpha, num)
    total_impact = impact_A + impact_B
    
    prob2 = np.mean(total_impact > point2)
    
    prob3 = np.mean((total_impact > point3) & (total_impact < point4))
    

    SLE = MEDIAN_t * prob2
    ALE = MEAN_d * SLE
    
    return (prob1, MEAN_t, MEDIAN_t, MEAN_d, VARIANCE_d, prob2, prob3, ALE)
# -- END OF YOUR CODERUNNER SUBMISSION CODE
