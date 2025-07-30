import numpy as np

def debug_task2(num, table, probs):
    """
    Debugging version of Task2 that prints intermediate values
    """
    # Extract conditional probabilities
    PX2, PX3, PX4, PX5, PY6, PY7 = probs
    
    print(f"\n--- DEBUG: Test with num={num} ---")
    print(f"Conditional probabilities: {probs}")
    
    # Convert to numpy array
    joint_counts = np.array(table)
    print(f"Joint counts:\n{joint_counts}")
    
    # Calculate joint probabilities
    joint_probs = joint_counts / num
    print(f"Joint probabilities:\n{joint_probs}")
    
    # Calculate marginal probabilities
    p_x = np.sum(joint_probs, axis=0)  # Sum over rows (Y) to get P(X)
    p_y = np.sum(joint_probs, axis=1)  # Sum over columns (X) to get P(Y)
    
    print(f"P(X=2,3,4,5): {p_x}")
    print(f"P(Y=6,7,8): {p_y}")
    
    # 1. Calculate prob1: P(3 ≤ X ≤ 4)
    prob1 = p_x[1] + p_x[2]
    print(f"prob1 = P(X=3) + P(X=4) = {p_x[1]} + {p_x[2]} = {prob1}")
    
    # 2. Calculate prob2: P(X+Y ≤ 10)
    prob2 = 0
    for i, y_val in enumerate([6, 7, 8]):
        for j, x_val in enumerate([2, 3, 4, 5]):
            if x_val + y_val <= 10:
                prob2 += joint_probs[i, j]
                print(f"  Adding P(X={x_val},Y={y_val}) = {joint_probs[i, j]} (sum = {x_val+y_val})")
    print(f"prob2 = P(X+Y<=10) = {prob2}")
    
    # 3. Calculate prob3 using different methods
    
    # Method 1: Direct calculation of p(T) and P(T∩Y=8)
    # Calculate P(T) = Σ P(T|X=x) * P(X=x)
    p_t_1 = 0
    print("\nCalculating P(T) using P(T|X):")
    for j, x_val in enumerate([2, 3, 4, 5]):
        p_t_x = probs[j] * p_x[j]
        p_t_1 += p_t_x
        print(f"  P(T|X={x_val}) * P(X={x_val}) = {probs[j]} * {p_x[j]} = {p_t_x}")
    print(f"P(T) = {p_t_1}")
    
    # Calculate P(T∩Y=8)
    p_t_and_y8_1 = 0
    print("\nCalculating P(T∩Y=8):")
    for j, x_val in enumerate([2, 3, 4, 5]):
        p_joint = probs[j] * joint_probs[2, j]
        p_t_and_y8_1 += p_joint
        print(f"  P(T|X={x_val}) * P(X={x_val},Y=8) = {probs[j]} * {joint_probs[2, j]} = {p_joint}")
    print(f"P(T∩Y=8) = {p_t_and_y8_1}")
    
    # Calculate P(Y=8|T) = P(T∩Y=8) / P(T)
    prob3_1 = p_t_and_y8_1 / p_t_1 if p_t_1 > 0 else 0
    print(f"Method 1 - P(Y=8|T) = P(T∩Y=8) / P(T) = {p_t_and_y8_1} / {p_t_1} = {prob3_1}")
    
    # Method 2: Using Bayes' theorem with P(T|Y=8)
    # Calculate P(T|Y=8) = Σ P(T|X=x) * P(X=x|Y=8)
    p_t_given_y8_2 = 0
    print("\nCalculating P(T|Y=8):")
    p_x_given_y8 = joint_probs[2, :] / p_y[2] if p_y[2] > 0 else np.zeros(4)
    print(f"P(X=2,3,4,5|Y=8): {p_x_given_y8}")
    
    for j, x_val in enumerate([2, 3, 4, 5]):
        p_t_xy = probs[j] * p_x_given_y8[j]
        p_t_given_y8_2 += p_t_xy
        print(f"  P(T|X={x_val}) * P(X={x_val}|Y=8) = {probs[j]} * {p_x_given_y8[j]} = {p_t_xy}")
    print(f"P(T|Y=8) = {p_t_given_y8_2}")
    
    # Calculate P(Y=8|T) = P(T|Y=8) * P(Y=8) / P(T)
    prob3_2 = (p_t_given_y8_2 * p_y[2]) / p_t_1 if p_t_1 > 0 else 0
    print(f"Method 2 - P(Y=8|T) = P(T|Y=8) * P(Y=8) / P(T) = {p_t_given_y8_2} * {p_y[2]} / {p_t_1} = {prob3_2}")
    
    # Method 3: Alternative approach
    # Calculate P(T) using P(T|Y=y) values for Y=6,7
    p_t_3 = PY6 * p_y[0] + PY7 * p_y[1]
    
    # For Y=8, we derive P(T|Y=8) using P(T|X) values
    p_t_y8 = 0
    for j in range(4):
        p_x_y8 = joint_probs[2, j] / p_y[2] if p_y[2] > 0 else 0
        if j == 0: p_t_y8 += PX2 * p_x_y8
        elif j == 1: p_t_y8 += PX3 * p_x_y8
        elif j == 2: p_t_y8 += PX4 * p_x_y8
        elif j == 3: p_t_y8 += PX5 * p_x_y8
    
    p_t_3 += p_t_y8 * p_y[2]  # Add P(T|Y=8) * P(Y=8)
    
    prob3_3 = (p_t_y8 * p_y[2]) / p_t_3 if p_t_3 > 0 else 0
    print(f"\nMethod 3 - P(T) using P(T|Y): {p_t_3}")
    print(f"Method 3 - P(Y=8|T) = {prob3_3}")
    
    # Return the values from the approach that worked for test case 3
    return (prob1, prob2, prob3_1)

# Test with all three test cases
# Test case 1
print("\n===== TEST CASE 1 =====")
num1 = 120
probs1 = [0.7, 0.6, 0.5, 0.63, 0.44, 0.36]
table1 = [[6, 10, 11, 9], [9, 12, 15, 8], [7, 14, 10, 9]]
results1 = debug_task2(num1, table1, probs1)
print(f"Final results: {results1}")

# Test case 2
print("\n===== TEST CASE 2 =====")
num2 = 200
probs2 = [0.5, 0.3, 0.6, 0.2, 0.4, 0.6]
table2 = [[14, 16, 20, 15], [17, 21, 20, 16], [12, 18, 15, 17]]
results2 = debug_task2(num2, table2, probs2)
print(f"Final results: {results2}")

# Test case 3 (passes)
print("\n===== TEST CASE 3 =====")
num3 = 80
probs3 = [0.3, 0.4, 0.3, 0.2, 0.1, 0.5]
table3 = [[9, 7, 8, 5], [10, 6, 3, 5], [6, 7, 5, 9]]
results3 = debug_task2(num3, table3, probs3)
print(f"Final results: {results3}")