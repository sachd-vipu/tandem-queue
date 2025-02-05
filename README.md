# Simulation of Tandem Queues with Arbitrary Network Topology

## Overview
This project implements a simulation of tandem queueing networks, supporting arbitrary queueing structures, different inter-arrival and service time distributions, and various performance evaluation techniques. The simulation is event-driven and can handle network partitioning with independent event stacks for each partition.

## Features
- **Generalized Queueing Network Simulation**: Supports arbitrary queue networks using a routing matrix for easy configuration.
- **Multiple Arrival and Service Distributions**: Includes exponential, Poisson, Erlang, Coxian, Lognormal, Beta, and more.
- **Partitioned Event Handling**: Allows for separate event stacks for different partitions, ensuring causal order through rollback mechanisms.
- **Performance Metrics**: Computes mean sojourn times, queue lengths, utilizations, and validates results using Little's Law and Jackson's Theorem.
- **Visualization and Statistical Analysis**: Generates plots for queue length variations, batch mean sojourn times, histograms, and theoretical vs. simulated comparisons.

## Installation
### Prerequisites
- Python 3.x
- Required libraries: `numpy`, `matplotlib`, `scipy`

Install dependencies using:
```bash
pip install numpy matplotlib scipy
```

## Usage
Run the simulation by executing the main script:
```bash
python simulation.py
```

## Configuration
### Network Topology
Define the queue network using:
```python
routing_matrix = {
    1: {2: 1.0},  # Node 1 routes to Node 2 with 100% probability
    2: {}          # Node 2 is an exit node
}
```
### Arrival and Service Time Distributions
Modify `arrival_distributions` and `service_rates_system` to configure inter-arrival and service time distributions:
```python
arrival_distributions = {
    1: 'exponential'
}
arrival_params = {
    1: {'mean': 1 / 5.0}  # Arrival rate of 5 jobs/sec
}
```

### Running Simulations
Modify the `main()` function to select different configurations:
```python
results = simulate_partition_queue_network(
    partitions=partitioned_nodes,
    routing_matrix=routing_matrix,
    external_arrival_rates=external_arrival_rates,
    arrival_distributions=arrival_distributions,
    arrival_params=arrival_params,
    simulation_period=900,
    seed=12981,
    confidence_level=0.99
)
```

## Validation & Analysis
### Little's Law Validation
```python
calculate_little_l(results, external_arrival_rates, mean_sojourn_time, confidence_level=0.99)
```
### Jackson Network Validation
```python
validate_with_jackson(results, arrival_rate=5.0, service_rates=[8.0, 8.0], num_queues=2)
```

### Visualization
- **Batch Means Plot:**
```python
plot_batch_means(results['sojourn_times'], batch_size=100)
```
- **Queue Length Over Time:**
```python
plt.step(times, queue_lengths, where='post')
```

## Results
### Exponential Arrivals and Service Time
- **Mean Sojourn Time**: 0.6811 seconds ± 0.0055 (99% CI)
- **Utilization**: ~0.625 per node (with μ=8, λ=5)
- **Queue Stability**: System remains stable when μ > λ.

### Non-Poisson Arrivals and Service Time
- Various arrival and service time distributions tested, including Weibull, Lognormal, and Hyperexponential.
- Systems may become unstable under heavy load or improper configurations.

## References
- [Wikipedia: Phase-Type Distributions](https://en.wikipedia.org/wiki/Phase-type_distribution)
- [Queueing Models Documentation](https://qmodels.readthedocs.io/en/latest/mm1.html)
- [Poisson and Exponential Distributions](https://neurophysics.ucsd.edu/courses/physics_171/exponential.pdf)

## Author
Vipul Sachdeva, CSE 517 Fall 2024

