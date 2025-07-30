# Cybersecurity Risk Modeling

Statistical analysis and optimization for cybersecurity risk assessment using probability distributions and Monte Carlo simulation.

### Task 1: Risk Assessment Analysis
- Built triangular distribution functions to model asset values (CDF, mean, median)
- Created Monte Carlo simulation for security flaw impacts using lognormal and Pareto distributions
- Calculated Annual Loss Expectancy (ALE) for cybersecurity risk planning
- Analyzed discrete probability patterns for security incident frequencies

```python
# Risk assessment with probability distributions
results = Task1(a, b, c, point1, number_set, prob_set, 1000, point2, mu, sigma, xm, alpha, point3, point4)
```

### Task 2: Joint Probability Analysis
- Implemented joint and marginal probability calculations for security scan workflows
- Applied Bayes' theorem to calculate conditional probabilities
- Modeled time requirements for security vulnerability detection and remediation

```python
# Security scan time analysis
prob_results = Task2(num, joint_table, conditional_probabilities)
```

### Task 3: Security Control Optimization
- Used linear regression to analyze historical security control effectiveness
- Built optimization algorithm to maximize security within budget constraints
- Balanced security improvement against maintenance costs

```python
# Optimize security control deployment
weights_b, weights_d, x_add = Task3(historical_data, effectiveness, maintenance, current_controls, costs, limits)
```

## Skills Used

- Statistical modeling with probability distributions
- Monte Carlo simulation methods
- Linear regression and optimization
- Bayesian probability analysis
- Risk assessment calculations

## Technologies

- Python 3.8+
- NumPy for mathematical operations
- Custom probability distribution implementations
- Linear algebra for regression analysis

## Academic Context

Coursework for SCC.363 Security and Risk at Lancaster University. Applies statistical methods to real cybersecurity risk management problems.

## Running the Code

```bash
pip install numpy
python task1.py
python task2.py  
python task3.py
```

## File Structure

- `task1.py` - Risk assessment and ALE calculations
- `task2.py` - Joint probability and Bayesian analysis
- `task3.py` - Security control optimization
- `task1test.py` - Test cases
- `test_task2.py` - Additional testing and debugging
