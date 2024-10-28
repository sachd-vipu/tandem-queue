import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import math
import random


DEBUG = False

def debug_print(s):   
    if DEBUG:
        print(s)


@dataclass(order=True,eq=True)
class Event:
    time: float
    event_type: str  # 'arrival', 'departure'
    job_id: int = field(compare=False, default=0)
    node_id: int = field(compare=False, default=0)
    next_event: ['Event'] = field(compare=False, default=None)
    prev_event: ['Event'] = field(compare=False, default=None)

class EventStack:
    def __init__(self):
        self.head: [Event] = None  
        self.tail: [Event] = None  

    def is_empty(self) -> bool:
        return self.head is None

    def insert_event(self, event: Event):

        if self.head is None:
            self.head = self.tail = event
            debug_print("first event: {event}")
        else:
            current = self.head
            while current and current.time <= event.time:
                current = current.next_event
            if current is None:
                self.tail.next_event = event
                event.prev_event = self.tail
                self.tail = event
                debug_print(f"Inserted at end: {event.job_id} -- {event.time}")
            elif current.prev_event is None:
                event.next_event = self.head
                self.head.prev_event = event
                self.head = event
                debug_print(f"Inserted event at start: {event.job_id} -- {event.time}")
            else:
                prev_node = current.prev_event
                prev_node.next_event = event
                event.prev_event = prev_node
                event.next_event = current
                current.prev_event = event
                debug_print(f"Inserted event in the middle: {event.job_id} -- {event.time}")

    def pop_event(self) -> [Event]:
        if self.head is None:
            return None
        event = self.head
        self.head = event.next_event
        if self.head:
            self.head.prev_event = None
        else:
            self.tail = None
        debug_print(f"Popped event: {event.job_id}")
        return event

class Queue:
    def __init__(self, node_id, service_distribution, service_params):
        self.node_id = node_id
        self.queue = deque()
        self.service_distribution = service_distribution
        self.service_params = service_params
        self.busy = False
        self.last_busy_time = None  
        self.total_service_time = 0.0 
        self.jobs_served = 0
        self.wait_times = []
        self.service_times = []

        
def generate_random_sample(distribution_type, params):
    
    U = np.random.uniform(0, 1)
    if distribution_type == 'exponential':
        #  {'mean': value}
        mean = params['mean']
        sample = -mean * np.log(1 - U)
        debug_print(f"Generated Exponential sample: {sample} with mean {mean}")
        return sample
    
    elif distribution_type == 'poisson':
        #  params = {'lambda': value}
        lamb_da = params['lambda']
        sample = np.random.poisson(lam)
        debug_print(f"Generated Poisson sample: {sample} with λ {lamb_da}")
        return sample
    
    elif distribution_type == 'erlang':
        # params = {'k': value, 'theta': value}
        k = params['k']
        theta = params['theta']
        samples = [-theta * np.log(1 - np.random.uniform(0, 1)) for _ in range(k)]
        sample = sum(samples)
        debug_print(f"Generated Erlang sample: {sample} with k={k}, theta={theta}")
        return sample


# Simulation
def simulate_queue_network(
    nodes: Dict[int, Queue],
    routing_matrix: Dict[int, Dict[int, float]],
    external_arrival_rates: Dict[int, float],
    arrival_distributions: Dict[int, str],
    arrival_params: Dict[int, Dict],
    total_jobs: int = 1000 ,
    warmup_period: float = 0.0,
    seed: Optional[int] = None
) -> Dict:


    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    event_stack = EventStack()
    time = 0.0
    job_id_counter = 0

    # All stats
    node_stats = {node_id: {'arrivals': 0, 'departures': 0, 'queue_lengths': [], 'times': []} for node_id in nodes}
    job_times = {}
    processed_jobs = 0
    total_sojourn_times = []
    job_start_times = {}
    job_end_times = {}

    # Schedule first job arrival
    for node_id, rate in external_arrival_rates.items():
        if rate > 0:
            inter_arrival_time = generate_random_sample(arrival_distributions[node_id], arrival_params[node_id])
            arrival_time = time + inter_arrival_time
            event = Event(arrival_time, 'arrival', job_id_counter, node_id)
            event_stack.insert_event(event)
            job_id_counter += 1
            debug_print(f"Scheduled initial arrival for Job {event.job_id} at Node {node_id} at time {arrival_time}")

    processed_jobs = 0

    '''
    Customer arrives
        |
        |-> Schdeule arrival of next customer
        |-> Check if customer can be served immediately
             |
             |-> If yes, schdeule departure
        |-> If departure is scheduled, that means customer is being served
    '''
    

    #Run simulation
    '''
     There are dangling jobs in the system after this loop , not reuquired for the simulation
       |-> TO-DO Copy same exact logic for but do not schedule next external arrival/collect stats -> while any(node.queue or node.busy for node in nodes.values()):
    '''
    while processed_jobs < total_jobs:
        
        event = event_stack.pop_event()
        if event is None:
            break
        time = event.time
        
        node_id = event.node_id
        node = nodes[node_id]
        stats = node_stats[node_id]

        if event.event_type == 'arrival':

            job_id = event.job_id
            if job_id not in job_times:
                job_times[job_id] = {
                    'arrival_times': {},
                    'departure_times': {}
                }

            job_times[job_id]['arrival_times'][node_id] = time
            node.queue.append(job_id)
            stats['arrivals'] += 1
            job_start_times[job_id] = time

            # If idle, start service
            if not node.busy:
                node.busy = True

                #  start tracking busy time
                if time >= warmup_period:
                    node.last_busy_time = time
                else:
                    node.last_busy_time = warmup_period  

                # Schedule departure and service time 
                service_time = generate_random_sample(node.service_distribution, node.service_params)
                departure_time = time + service_time
                departure_event = Event(departure_time, 'departure', job_id, node_id)
                event_stack.insert_event(departure_event)
                node.service_times.append(service_time)
                debug_print(f"Job {job_id} started at Node {node_id}; time - {time}, set to leave at - {departure_time}")

            # Schedule next external arrival
            if node_id in external_arrival_rates:
                rate = external_arrival_rates[node_id]
                if rate > 0:
                    inter_arrival_time = generate_random_sample(arrival_distributions[node_id], arrival_params[node_id])
                    next_arrival_time = time + inter_arrival_time
                    new_event = Event(next_arrival_time, 'arrival', job_id_counter, node_id)
                    event_stack.insert_event(new_event)
                    debug_print(f"Next arrival for Job {job_id_counter} at Node - {node_id}; time - {next_arrival_time}")
                    job_id_counter += 1

        elif event.event_type == 'departure':

            job_id = event.job_id
            node.queue.popleft()
            stats['departures'] += 1
            job_times[job_id]['departure_times'][node_id] = time

            # Route to next node / exit
            routing_probs = routing_matrix.get(node_id, {})
            if routing_probs:
                next_node_ids = list(routing_probs.keys())
                next_probs = list(routing_probs.values())
                next_node_id = np.random.choice(next_node_ids, p=next_probs)
                next_event = Event(time, 'arrival', job_id, next_node_id)
                event_stack.insert_event(next_event)
                debug_print(f"Job {job_id} routed to Node {next_node_id} at time {time}")
            else:
                # Exit the system
                processed_jobs += 1
                job_end_times[job_id] = time
                first_arrival = min(job_times[job_id]['arrival_times'].values())
                sojourn_time = time - first_arrival
                total_sojourn_times.append(sojourn_time)
                debug_print(f"Job {job_id} exited the system at time {time}, sojourn time was {sojourn_time}")

            # Collect service times -> Also mark node as idle, set busy time to None
            if node.busy and node.last_busy_time is not None:
                node.total_service_time += time - node.last_busy_time
                node.busy = False
                node.last_busy_time = None

            # Start service for next job
            if node.queue:
                next_job_id = node.queue[0]
                node.busy = True
                node.last_busy_time = max(time, warmup_period)
                service_time = generate_random_sample(node.service_distribution, node.service_params)
                departure_time = time + service_time
                departure_event = Event(departure_time, 'departure', next_job_id, node_id)
                event_stack.insert_event(departure_event)
                node.service_times.append(service_time)
                debug_print(f"Job {next_job_id} started service at Node {node_id} at time {time}, will depart at {departure_time}")
            else:
                node.busy = False
                node.last_busy_time = None
                debug_print(f"Node {node_id} idle at time - {time}")

            stats['queue_lengths'].append(len(node.queue))
            stats['times'].append(time)

    # for node_id, node in nodes.items():
    #     if node.busy and node.last_busy_time is not None:
    #         print(f"Node {node_id} still busy at time {time}")
    #         node.total_service_time += time - node.last_busy_time
    #         node.busy = False
    #         node.last_busy_time = None


    #  Performance metrics 
    overall_mean_sojourn_time = np.mean(total_sojourn_times) if total_sojourn_times else 0.0
    confidence_interval = 1.96 * np.std(total_sojourn_times) / math.sqrt(len(total_sojourn_times)) if total_sojourn_times else 0.0
    utilizations = {}
    total_simulation_time = time - warmup_period if time > warmup_period else time
    for node_id, node in nodes.items():
        utilization = node.total_service_time / total_simulation_time if total_simulation_time > 0 else 0.0
        utilizations[node_id] = utilization

    return {
        'mean_sojourn_time': overall_mean_sojourn_time,
        'confidence_interval': confidence_interval,
        'node_stats': node_stats,
        'utilizations': utilizations,
        'sojourn_times': total_sojourn_times,
        'simulation_time': time
    }


# Main 
def main():
    # network topology mu1 = mu2 = 9.0 (mean service time = 1/mu )

    
    nodes = {
        1: Queue(1, 'exponential', {'mean': 1/6.0}),  
        2: Queue(2, 'exponential', {'mean': 1/6.0})   
    }

    # Routing matrix: Node 1 -> Node 2 -> Exit
    #{ from_node: {to_node: probability} }
    routing_matrix = {
        1: {2: 1.0}, 
        2: {}         
    }

    # Define external arrival rates: Node 1 receives external arrivals
    external_arrival_rates = {
        1: 4.0  # Arrival rate lambda = 4.0 per hour
    }

    # Define arrival distributions and parameters
    arrival_distributions = {
        1: 'exponential'
    }

    arrival_params = {
        1: {'mean': 1 / external_arrival_rates[1]}  # Mean inter-arrival time = 0.2 seconds
    }

    # Simulation parameters
    total_jobs = 10000
    warmup_period = 100  
    seed = 45698  

    # Run simulation
    results = simulate_queue_network(
        nodes=nodes,
        routing_matrix=routing_matrix,
        external_arrival_rates=external_arrival_rates,
        arrival_distributions=arrival_distributions,
        arrival_params=arrival_params,
        total_jobs=total_jobs,
        warmup_period=warmup_period,
        seed=seed
    )

    # Extract results
    mean_sojourn_time = results['mean_sojourn_time']
    confidence_interval = results['confidence_interval']
    utilizations = results['utilizations']
    simulation_time = results['simulation_time']
    sojourn_times = results['sojourn_times']

    print(f"Simulation finished at time: {simulation_time:.4f} seconds")
    print(f"Mean Sojourn Time: {mean_sojourn_time:.4f} ± {confidence_interval:.4f} seconds")
    for node_id, utilization in utilizations.items():
        print(f"Utilization of Node {node_id}: {utilization:.4f}")
   
    plt.figure(figsize=(10, 6))
    plt.hist(sojourn_times, bins='auto', edgecolor='black')
    plt.xlabel('Sojourn Time (seconds)')
    plt.ylabel('Number of Jobs')
    plt.title('Histogram of Sojourn Times')
    plt.grid(True)
    plt.show()
    
    # Plotting queue lengths over time for each node
    # fig, axs = plt.subplots(len(results['node_stats']), 1, figsize=(10, 6*len(results['node_stats'])))
    # for i, (node_id, stats) in enumerate(results['node_stats'].items()):
    #     times = stats['times']
    #     queue_lengths = stats['queue_lengths']
    #     if times and queue_lengths:
    #         axs[i].step(times, queue_lengths, where='post')
    #         axs[i].set_xlabel('Time')
    #         axs[i].set_ylabel(f'Queue Length at Node {node_id}')
    #         axs[i].set_title(f'Queue Length Over Time at Node {node_id}')
    #         axs[i].grid(True)

    # plt.tight_layout()
    # plt.show()

    # # Plot mean sojourn time with confidence interval
    # plt.figure(figsize=(6, 6))
    # plt.bar(['Mean Sojourn Time'], [mean_sojourn_time], yerr=[confidence_interval], capsize=10, color='skyblue')
    # plt.ylabel('Time (seconds)')
    # plt.title('Mean Sojourn Time with 95% Confidence Interval')
    # plt.show()

    # Validation with Jackson's Theorem
    validate_with_jackson(simulation=results,arrival_rate=external_arrival_rates[1],service_rates=[6.0, 6.0],num_queues=2 )

    calculate_little_l(results, external_arrival_rates, mean_sojourn_time)



    # Calculate Little's L
def calculate_little_l(results: Dict, external_arrival_rates: Dict[int, float], mean_sojourn_time: float) -> None:
 
    lambda_system = external_arrival_rates[1] 
    L_little = lambda_system * mean_sojourn_time

    # Calculate simulated average number of jobs in the system
    # Assuming the system consists of two queues, sum their average queue lengths
    average_L_simulated = 0.0
    for node_id, stats in results['node_stats'].items():
        average_L_simulated += np.mean(stats['queue_lengths'])  # Average queue length per node

    print(f"Little's L (λW): {L_little:.4f}")
    print(f"Simulated Average Number of Jobs in System: {average_L_simulated:.4f}")
    print(f"Little's L (λW) / Simulated Average Number of Jobs in System: {L_little / average_L_simulated:.4f}")
    print(f"Percentage Error from simulated average Jobs: {(abs(L_little - average_L_simulated ) / average_L_simulated ) * 100:.4f}%")


# Validation Function - Jackson's Theorem 
def validate_with_jackson(simulation: Dict, arrival_rate: float, service_rates: List[float], num_queues: int):
 
    # effective arrival rate lambda_i = lambda for all queues -in Poisson arrivals and exponential services
    theoretical_wait_times = [1 / (mu - arrival_rate) for mu in service_rates]
    theoretical_total_sojourn = sum(theoretical_wait_times)

    # Simulated mean sojourn time
    simulated_total_sojourn = simulation['mean_sojourn_time']

    print("\n--- Validation with Jackson's Theorem ---")
    for i in range(num_queues):
        print(f"Queue {i+1}: Theoretical Wait time = {theoretical_wait_times[i]:.4f} seconds")

    print(f"Theoretical Total Sojourn Time: {theoretical_total_sojourn:.4f} seconds")
    print(f"Simulated Total Sojourn Time: {simulated_total_sojourn:.4f} seconds")


    service_rates = np.linspace(5, 32, num=22)  # Vary service rate from 5 to 15 jobs per unit time
    mean_sojourn_times = []
    confidence_intervals =[]
    for rate in service_rates:
        nodes = {
            1: Queue(1, 'exponential', {'mean': 1/rate}),  # Varying service rate
            2: Queue(2, 'exponential', {'mean': 1/rate})  # Constant service rate for Node 2
        }
        results = simulate_queue_network(
            nodes,
            routing_matrix={
                1: {2: 1.0},
                2: {}
            },
            external_arrival_rates={1: 4.0},  # Constant arrival rate
            arrival_distributions={1: 'exponential'},
            arrival_params={1: {'mean': 1/4.0}},
            total_jobs=10000,
            warmup_period=100  # Using a warmup period to stabilize
        )
        mean_sojourn_times.append(results['mean_sojourn_time'])
        confidence_intervals.append(results['confidence_interval'])

    plt.figure()
    plt.plot(service_rates, mean_sojourn_times, marker='o')
    plt.errorbar(service_rates, mean_sojourn_times, yerr=confidence_intervals, fmt='-o', capsize=5)
    plt.xlabel('Service Rate (jobs/unit time)')
    plt.ylabel('Mean Sojourn Time')
    plt.title('Mean Sojourn Time vs. Service Rate')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()


