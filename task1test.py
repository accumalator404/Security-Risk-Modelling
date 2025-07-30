import numpy as np
import random

np.random.seed(42)
random.seed(42)

from task1 import Task1

a, b, c = 100, 500, 300
point1 = 250
number_set = [0, 1, 2, 3, 4]
prob_set = [0.1, 0.2, 0.4, 0.2, 0.1]
num = 1000
point2 = 200
mu, sigma = 4, 0.5
xm, alpha = 50, 2
point3, point4 = 150, 300

# Run Task1 function
results = Task1(a, b, c, point1, number_set, prob_set, num, point2, 
               mu, sigma, xm, alpha, point3, point4)

print ("results:", results)