import numpy as np
# The Task3 function solves a problem related to optimising the deploymoent of security controls 
# in a company's network.
# The goal here is to improve safeguard effect while keeping a maintenance load.
# So how much effort is needed to main the contrls within a specified limit.

# Inputs: 
# x - historical data about how many security controls were used in the past
# y - historical data about how effective the controls were in safeguarding 
# z - historical data about how much maintenance was required for the controls
# x_initial - The current number of each type of security control being used
# - c: The cost of adding each type of security control.
# - x_bound: The maximum number of each type of security control that can be deployed.
# - se_bound: The minimum level of safeguard effect the company wants to achieve.
# - ml_bound: The maximum level of maintenance load the company can handle.

# 1. The function uses historical data to figure how different numbers of controls affect
#   the safeguard effect and maintenance load. This is done using linear regression, to find the patterns in the data.
# 
# 2. Using the patterns from step 1 the function calclates the current safeguard effect and maintanence loa,
#     based on the controls already in place (x_initial)

# 3. If current safeguard is less than the desired level (se_bound), the function figures out how to add more controls
#    in a cost-effective way. It will prioritise controls that give the most safeguard effect with the least cost.

# 4. In the midst of the process the function ensures the maintenance load does not exceed the company's limit (ml_bound).
#      It also makes sure not to add more controls than allowed (x_bound)

# 5. The functio will then reutrn the patterns it's found (weights_b and weights_d) and the number of addtional 
#    controls to add (x_add) to meet the company goals

def Task3(x, y, z, x_initial, c, x_bound, se_bound, ml_bound):
    
    historical_data = np.array(x).T
    
    data_with_intercept = np.column_stack([np.ones(historical_data.shape[0]), historical_data])
    
    weights_b = np.linalg.inv(data_with_intercept.T.dot(data_with_intercept)).dot(data_with_intercept.T).dot(y)
    
    weights_d = np.linalg.inv(data_with_intercept.T.dot(data_with_intercept)).dot(data_with_intercept.T).dot(z)
    costs = np.array(c)
    
    safeguard_coeffs = np.array([weights_b[1], weights_b[2], weights_b[3], weights_b[4]])
    current_safeguard = weights_b[0]
    for i in range(4):
        current_safeguard += safeguard_coeffs[i] * x_initial[i]
    
    maintenance_coeffs = np.array([weights_d[1], weights_d[2], weights_d[3], weights_d[4]])
    current_maintenance = weights_d[0]
    for i in range(4):
        current_maintenance += maintenance_coeffs[i] * x_initial[i]
    
    remaining_capacity = np.array(x_bound) - np.array(x_initial)
    
    
    x_add = np.zeros(4)
    
    if current_safeguard < se_bound:
        effectiveness = safeguard_coeffs / costs
        
        sorted_indices = np.argsort(-effectiveness)
        
        safeguard_needed = se_bound - current_safeguard
        
        for idx in sorted_indices:
            max_allowed = remaining_capacity[idx]
            maintenance_per_unit = maintenance_coeffs[idx]
            safeguard_per_unit = safeguard_coeffs[idx]
            if current_maintenance + maintenance_per_unit * max_allowed <= ml_bound:
                units_to_add = min(max_allowed, safeguard_needed / safeguard_per_unit)
            else:
                available_maintenance = ml_bound - current_maintenance
                units_to_add = min(max_allowed, available_maintenance / maintenance_per_unit)
                
            x_add[idx] = units_to_add
            
            current_safeguard += safeguard_per_unit * units_to_add
            current_maintenance += maintenance_per_unit * units_to_add
            safeguard_needed -= safeguard_per_unit * units_to_add
            if current_safeguard >= se_bound:
                break
    
    return (weights_b, weights_d, x_add)