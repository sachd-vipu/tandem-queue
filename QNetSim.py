import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import math
import random
import copy
import scipy.stats 

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

    # track intiator stack for this event in case of partition
    source_stack_id: int = field(compare=False, default=0)  

class EventStack:
    def __init__(self, stack_id: Optional[int]=1):
        self.head: [Event] = None  
        self.tail: [Event] = None

        # For netowrk partition and separate evenstack logic  
        self.current_time = 0.0  
        self.stack_id = stack_id  
        self.state_history = []  

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

    # Methods for handling partitions and rollback
    def save_state(self, state):
        self.state_history.append((self.current_time, copy.deepcopy(state)))

    def rollback(self, rollback_time):
        # Revert back to last saved state and remove futurre states
        for i in reversed(range(len(self.state_history))):
            time, state = self.state_history[i]
            if time <= rollback_time:
                self.current_time = time
                self.state_history = self.state_history[:i+1]
                return state
        return None

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
        sample = np.random.poisson(lamb_da)
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

    elif distribution_type == 'hyperexponential':
        # params = {'p': [value1, value2], 'means': [value1, value2]}
        p = params['p']
        means = params['means']
        chosen_mean = np.random.choice(means, p=p)
        sample = -chosen_mean * np.log(1 - U)
        debug_print(f"Generated Hyperexponential sample: {sample} with p={p}, means={means}")
        return sample

    elif distribution_type == 'deterministic':
        # params = {'value': value}
        sample = params['value']
        debug_print(f"Generated Deterministic sample: {sample}")
        return sample

    elif distribution_type == 'uniform':
        # params = {'low': value, 'high': value}
        sample = np.random.uniform(params['low'], params['high'])
        debug_print(f"Generated Uniform sample: {sample}")
        return sample

    elif distribution_type == 'weibull':
        # params = {'shape': value, 'scale': value}
        sample = np.random.weibull(params['shape']) * params['scale']
        debug_print(f"Generated Weibull sample: {sample}")
        return sample


def find_z_critical(confidence_level):
    alpha = 1 - confidence_level
    z_critical = scipy.stats.norm.ppf(1 - alpha / 2)
    return z_critical

# Simulation
def simulate_queue_network(
    nodes: Dict[int, Queue],
    routing_matrix: Dict[int, Dict[int, float]],
    external_arrival_rates: Dict[int, float],
    arrival_distributions: Dict[int, str],
    arrival_params: Dict[int, Dict],
    total_jobs: int = 1000 ,
    warmup_period: float = 0.0,
    simulation_period: float = 100.0,
    time_interval: float = 1,
    seed: Optional[int] = None,
    confidence_level:Optional[float]= 0.95
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

    total_simulation_time = simulation_period + warmup_period
   
    time_next_record = warmup_period + time_interval
    time_records = []
    avg_queue_lengths = {node_id: [] for node_id in nodes}
    utilizations_over_time = {node_id: [] for node_id in nodes}
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
    #while processed_jobs < total_jobs:
    while time < total_simulation_time:    
        event = event_stack.pop_event()
        if event is None:
            break
        time = event.time

        if time > total_simulation_time :
            break
        
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

            # # Collect service times -> Also mark node as idle, set busy time to None
            # if node.busy and node.last_busy_time is not None:
            #     node.total_service_time += time - node.last_busy_time
            #     node.busy = False
            #     node.last_busy_time = None

            # Start service for next job
            if node.queue:
                next_job_id = node.queue[0]
                service_time = generate_random_sample(node.service_distribution, node.service_params)
                departure_time = time + service_time
                departure_event = Event(departure_time, 'departure', next_job_id, node_id)
                event_stack.insert_event(departure_event)
                node.service_times.append(service_time)
                debug_print(f"Job {next_job_id} started service at Node {node_id} at time {time}, will depart at {departure_time}")
            else:
                # Node becomes idle
                if node.busy and node.last_busy_time is not None:
                    if time >= node.last_busy_time:
                        node.total_service_time += time - node.last_busy_time
                    node.busy = False
                    node.last_busy_time = None
                    debug_print(f"Node {node_id} idle at time - {time}")

            if time >= warmup_period:
                node_stats[node_id]['queue_lengths'].append(len(node.queue))
                node_stats[node_id]['times'].append(time)

                if time >= time_next_record:
                    time_records.append(time)
                    for node_id, node in nodes.items():
                        avg_queue_lengths[node_id].append(len(node.queue))
                        if node.busy and node.last_busy_time is not None:
                            utilization = (time - node.last_busy_time) / (time - time_records[-2] if len(time_records) > 1 else time_interval)
                        else:
                            utilization = 0.0
                        utilizations_over_time[node_id].append(utilization)
                    time_next_record += time_interval

    # Dangling Jobs           
    # for node_id, node in nodes.items():
    #     if node.busy and node.last_busy_time is not None:
    #         print(f"Node {node_id} still busy at time {time}")
    #         node.total_service_time += time - node.last_busy_time
    #         node.busy = False
    #         node.last_busy_time = None


    #  Performance metrics 
    overall_mean_sojourn_time = np.mean(total_sojourn_times) if total_sojourn_times else 0.0
    confidence_interval = find_z_critical(confidence_level) * np.std(total_sojourn_times) / math.sqrt(len(total_sojourn_times)) if total_sojourn_times else 0.0
    utilizations = {}
    total_simulation_time = time - warmup_period if time > warmup_period else time
    for node_id, node in nodes.items():
        if node.busy and node.last_busy_time is not None:
            if time >= node.last_busy_time:
                node.total_service_time += time - node.last_busy_time
            node.busy = False
            node.last_busy_time = None
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


def simulate_partition_queue_network(
    partitions: Dict[int, Dict],  
    routing_matrix: Dict[int, Dict[int, float]],
    external_arrival_rates: Dict[int, float],
    arrival_distributions: Dict[int, str],
    arrival_params: Dict[int, Dict],
    total_jobs: int = 1000 ,
    warmup_period: float = 0.0,
    simulation_period: float = 100.0,
    time_interval: float = 1,
    seed: Optional[int] = None,
    confidence_level:Optional[float]= 0.95

) -> Dict:

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    job_id_counter = 0

    # All stats
    node_stats = {node_id: {'arrivals': 0, 'departures': 0, 'queue_lengths': [], 'times': []}
                  for partition in partitions.values() for node_id in partition['nodes']}
    job_times = {}
    processed_jobs = 0
    total_sojourn_times = []
    job_start_times = {}
    job_end_times = {}

    # Schedule first job arrival for each partition
    for partition_id, partition in partitions.items():
        event_stack = partition['event_stack']
        nodes = partition['nodes']
        time = event_stack.current_time

        for node_id, rate in external_arrival_rates.items():
            if node_id in nodes and rate > 0:
                inter_arrival_time = generate_random_sample(arrival_distributions[node_id], arrival_params[node_id])
                arrival_time = time + inter_arrival_time
                event = Event(arrival_time, 'arrival', job_id_counter, node_id, source_stack_id=partition_id)
                event_stack.insert_event(event)
                job_id_counter += 1
                debug_print(f"Scheduled initial arrival for Job {event.job_id} at Node {node_id} at time {arrival_time}")

    total_simulation_time = simulation_period + warmup_period

    # Main simulation loop
    while True:
        # Find the next event across all partitions
        next_events = []
        for partition in partitions.values():
            if partition['event_stack'].head:
                next_events.append((partition['event_stack'].head.time, partition['event_stack']))
        if not next_events:
            break
        next_time, event_stack = min(next_events, key=lambda x: x[0])

        # Process events in the selected event stack
        event = event_stack.pop_event()
        if event is None:
            continue

        # Check for rollback
        if event.time < event_stack.current_time:
        
            print(f"Rollback in Stack {event_stack.stack_id} to time {event.time}")
            state = event_stack.rollback(event.time)
            if state is None:
                nodes = partitions[event_stack.stack_id]['nodes']
                for node in nodes.values():
                    node.queue.clear()
                    node.busy = False
                    node.last_busy_time = None
            else:
                partitions[event_stack.stack_id]['nodes'] = state['nodes']
            # Re-insert 
            event_stack.insert_event(event)
            continue

        event_stack.current_time = event.time
        time = event.time

        if time > total_simulation_time:
            break

        node_id = event.node_id
        node = None
        for partition in partitions.values():
            if node_id in partition['nodes']:
                node = partition['nodes'][node_id]
                break
        if node is None:
            continue

        stats = node_stats[node_id]

        # Save state before processing the event
        state = {'nodes': partitions[event_stack.stack_id]['nodes']}
        event_stack.save_state(state)

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
                node.current_job_id = job_id

                # Schedule departure and service time 
                service_time = generate_random_sample(node.service_distribution, node.service_params)
                departure_time = time + service_time
                departure_event = Event(departure_time, 'departure', job_id, node_id, source_stack_id=event_stack.stack_id)
                event_stack.insert_event(departure_event)
                node.service_times.append(service_time)
                debug_print(f"Job {job_id} started at Node {node_id}; time - {time}, set to leave at - {departure_time}")

            # Schedule next external arrival
            if node_id in external_arrival_rates:
                rate = external_arrival_rates[node_id]
                if rate > 0:
                    inter_arrival_time = generate_random_sample(arrival_distributions[node_id], arrival_params[node_id])
                    next_arrival_time = time + inter_arrival_time
                    new_event = Event(next_arrival_time, 'arrival', job_id_counter, node_id, source_stack_id=event_stack.stack_id)
                    event_stack.insert_event(new_event)
                    debug_print(f"Next arrival for Job {job_id_counter} at Node - {node_id}; time - {next_arrival_time}")
                    job_id_counter += 1

        elif event.event_type == 'departure':

            job_id = event.job_id
            if node.queue and node.queue[0] == job_id:
                node.queue.popleft()
            stats['departures'] += 1
            job_times[job_id]['departure_times'][node_id] = time

            # Route to next node / exit
            routing_probs = routing_matrix.get(node_id, {})
            if routing_probs:
                next_node_ids = list(routing_probs.keys())
                next_probs = list(routing_probs.values())
                next_node_id = np.random.choice(next_node_ids, p=next_probs)
                
                # Determine next partition where routed to node is present
                for partition_id, partition in partitions.items():
                    if next_node_id in partition['nodes']:
                        next_partition_id = partition_id
                        next_event_stack = partition['event_stack']
                        break
                # communication delay
                communication_delay = np.random.uniform(-1.0, 1.0)  
                arrival_time = time + communication_delay
                arrival_time = max(arrival_time, 0.0)
                next_event = Event(arrival_time, 'arrival', job_id, next_node_id, source_stack_id=next_partition_id)
                if arrival_time < next_event_stack.current_time:
                    print(f"Rollback in Stack {next_event_stack.stack_id} due to incoming event at time {arrival_time}")
                    state = next_event_stack.rollback(arrival_time)
                    if state is None:
                        # Handle initial state
                        nodes = partitions[next_event_stack.stack_id]['nodes']
                        for node in nodes.values():
                            node.queue.clear()
                            node.busy = False
                            node.last_busy_time = None
                    else:
                        partitions[next_event_stack.stack_id]['nodes'] = state['nodes']
                next_event_stack.insert_event(next_event)
                debug_print(f"Job {job_id} routed to Node {next_node_id} at time {arrival_time}")
            else:
                # Exit the system
                processed_jobs += 1
                job_end_times[job_id] = time
                first_arrival = min(job_times[job_id]['arrival_times'].values())
                sojourn_time = time - first_arrival
                total_sojourn_times.append(sojourn_time)
                debug_print(f"Job {job_id} exited the system at time {time}, sojourn time was {sojourn_time}")

            # Start service for next job
            if node.queue:
                next_job_id = node.queue[0]
                node.current_job_id = next_job_id
                service_time = generate_random_sample(node.service_distribution, node.service_params)
                departure_time = time + service_time
                departure_event = Event(departure_time, 'departure', next_job_id, node_id, source_stack_id=event_stack.stack_id)
                event_stack.insert_event(departure_event)
                node.service_times.append(service_time)
                debug_print(f"Job {next_job_id} started service at Node {node_id} at time {time}, will depart at {departure_time}")
            else:
                # Node becomes idle
                node.busy = False
                node.current_job_id = None
                debug_print(f"Node {node_id} idle at time - {time}")

            if time >= warmup_period:
                node_stats[node_id]['queue_lengths'].append(len(node.queue))
                node_stats[node_id]['times'].append(time)

    #  Performance metrics 
    overall_mean_sojourn_time = np.mean(total_sojourn_times) if total_sojourn_times else 0.0
    confidence_interval = find_z_critical(confidence_level) * np.std(total_sojourn_times) / math.sqrt(len(total_sojourn_times)) if total_sojourn_times else 0.0
    utilizations = {}
    total_simulation_time = time - warmup_period if time > warmup_period else time
    for partition in partitions.values():
        nodes = partition['nodes']
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


def plot_batch_means(data, batch_size,confidence_interval:Optional[float]=0.95):
    num_batches = len(data) // batch_size
    batch_means = []
    batch_conf_intervals = []
    for i in range(num_batches):
        batch = data[i*batch_size:(i+1)*batch_size]
        batch_mean = np.mean(batch)
        batch_means.append(batch_mean)
        z_critical = find_z_critical(confidence_level)
        conf_interval = z_score * np.std(batch) / math.sqrt(len(batch))
        batch_conf_intervals.append(conf_interval)
    plt.figure()
    plt.errorbar(range(len(batch_means)), batch_means, yerr=batch_conf_intervals, fmt='o-', ecolor='r', capsize=5)
    plt.xlabel('Batch Number')
    plt.ylabel('Batch Mean Sojourn Time')
    plt.title('Batch Means Over Time with 95% Confidence Intervals')
    plt.grid(True)
    plt.show()

    # Calculate Little's L
def calculate_little_l(results: Dict, external_arrival_rates: Dict[int, float], mean_sojourn_time: float) -> None:
 
    lambda_system = external_arrival_rates[1] 
    L_little = lambda_system * mean_sojourn_time

    average_L_simulated = 0.0
    for node_id, stats in results['node_stats'].items():
        average_L_simulated += np.mean(stats['queue_lengths'])  # Average queue length per node

    print(f"Little's L (λW): {L_little:.4f}")
    print(f"Simulated Average Number of Jobs in System: {average_L_simulated:.4f}")
    print(f"Little's L (λW) / Simulated Average Number of Jobs in System: {L_little / average_L_simulated:.4f}")
    print(f"Percentage Error from simulated average Jobs: {(abs(L_little - average_L_simulated ) / average_L_simulated ) * 100:.4f}%")
    print(f"Percentage Error: {(abs(L_little - average_L_simulated ) / average_L_simulated ) * 100:.2f}%")

def validate_with_jackson(simulation: Dict, arrival_rate: float, service_rates: List[float], num_queues: int):
    # Calculate theoretical performance metrics
    theoretical_wait_times = []
    theoretical_queue_lengths = []
    theoretical_utilizations = []
    theoretical_sojourn_times = []
    for mu in service_rates:
        rho = arrival_rate / mu
        if rho >= 1:
            print(f"System is unstable at service rate μ = {mu}")
            theoretical_utilizations.append(rho)
            theoretical_queue_lengths.append(float('inf'))
            theoretical_wait_times.append(float('inf'))
            theoretical_sojourn_times.append(float('inf'))
        else:
            # Utilization
            theoretical_utilizations.append(rho)
            # Average number in system (L) for M/M/1
            L = rho / (1 - rho)
            theoretical_queue_lengths.append(L)
            # Average waiting time in system (W)
            W = 1 / (mu - arrival_rate)
            theoretical_wait_times.append(W)
            # Sojourn time at node (same as W here since service time is exponential)
            theoretical_sojourn_times.append(W)
    theoretical_total_sojourn = sum(theoretical_sojourn_times)

    # Simulated performance metrics
    simulated_total_sojourn = simulation['mean_sojourn_time']
    simulated_utilizations = simulation['utilizations']
    simulated_avg_queue_lengths = []
    for node_id in range(1, num_queues + 1):
        stats = simulation['node_stats'][node_id]
        avg_queue_length = np.mean(stats['queue_lengths']) if stats['queue_lengths'] else 0.0
        simulated_avg_queue_lengths.append(avg_queue_length)
    simulated_utilizations_list = [simulated_utilizations[node_id] for node_id in range(1, num_queues + 1)]

    # Output comparisons
    print("\n--- Comparison of Utilization, Average Queue Length, and Sojourn Time with Jackson's Theorem ---")
    for i in range(num_queues):
        print(f"Node  {i+1}:")
        print(f"  Theoretical Utilization: {theoretical_utilizations[i]:.4f}")
        print(f"  Simulated Utilization: {simulated_utilizations_list[i]:.4f}")
        print(f"  Theoretical Average Queue Length: {theoretical_queue_lengths[i]:.4f}")
        print(f"  Simulated Average Queue Length: {simulated_avg_queue_lengths[i]:.4f}")
        print(f"  Theoretical Sojourn Time: {theoretical_sojourn_times[i]:.4f} seconds")
    print(f"Theoretical Total Sojourn Time: {theoretical_total_sojourn:.4f} seconds")
    print(f"Simulated Total Sojourn Time: {simulated_total_sojourn:.4f} seconds")
    percentage_error = abs(simulated_total_sojourn - theoretical_total_sojourn) / theoretical_total_sojourn * 100
    print(f"Percentage Error in Sojourn Time: {percentage_error:.2f}%")

    # Plot comparison of sojourn times
    nodes = [f"Node {i+1}" for i in range(num_queues)]
    x = np.arange(len(nodes))
    width = 0.35  

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, theoretical_sojourn_times, width, label='Theoretical')
    rects2 = ax.bar(x + width/2, [simulated_total_sojourn / num_queues]*num_queues, width, label='Simulated')

    ax.set_ylabel('Sojourn Time (seconds)')
    ax.set_title('Sojourn Time per Node')
    ax.set_xticks(x)
    ax.set_xticklabels(nodes)
    ax.legend()
    plt.show()

# Validation Function - Jackson's Theorem 
def validate_with_jackson2(simulation: Dict, arrival_rate: float, service_rates: List[float], num_queues: int = 2) -> None:

    theoretical_wait_times = []
    theoretical_queue_lengths = []
    theoretical_utilizations = []
    theoretical_sojourn_times = []

    for mu in service_rates:
        rho = arrival_rate / mu
        if rho >= 1:
            print(f"System is unstable at service rate {mu}")
            theoretical_utilizations.append(rho)
            theoretical_queue_lengths.append(float('inf'))
            theoretical_wait_times.append(float('inf'))
            theoretical_sojourn_times.append(float('inf'))
        else:
            theoretical_utilizations.append(rho)
            L = rho / (1 - rho)
            theoretical_queue_lengths.append(L)
            W = 1 / (mu - arrival_rate)
            theoretical_wait_times.append(W)

    theoretical_total_sojourn = sum(theoretical_wait_times)

    simulated_total_sojourn = simulation['mean_sojourn_time']
    simulated_utilizations = simulation['utilizations']
    simulated_avg_queue_lengths = []
    for node_id in range(1, num_queues + 1):
        stats = simulation['node_stats'][node_id]
        avg_queue_length = np.mean(stats['queue_lengths']) if stats['queue_lengths'] else 0.0
        simulated_avg_queue_lengths.append(avg_queue_length)
    simulated_utilizations_list = [simulated_utilizations[node_id] for node_id in range(1, num_queues + 1)]
    percentage_error = abs(simulated_total_sojourn - theoretical_total_sojourn) / theoretical_total_sojourn * 100

    
    print(f"Theoretical Total Sojourn Time: {theoretical_total_sojourn:.4f} seconds")
    print(f"Simulated Total Sojourn Time: {simulated_total_sojourn:.4f} seconds")
    print(f"Percentage Error in Sojourn Time: {percentage_error:.2f}%")
    
   
    # Plot comparison of sojourn times
    nodes = [f"Node {i+1}" for i in range(num_queues)]
    x = np.arange(len(nodes))
    width = 0.35  # Width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, theoretical_sojourn_times, width, label='Theoretical')
    rects2 = ax.bar(x + width/2, [simulated_total_sojourn / num_queues]*num_queues, width, label='Simulated')

    ax.set_ylabel('Sojourn Time (seconds)')
    ax.set_title('Sojourn Time per Node')
    ax.set_xticks(x)
    ax.set_xticklabels(nodes)
    ax.legend()

    plt.show()

def run_simulations_and_plot(service_rates, routing_matrix, external_arrival_rates, arrival_distributions, arrival_params, total_jobs, warmup_period, simulation_period, time_interval, seed):
    all_results = {}
    mean_sojourn_times = []
    confidence_intervals = []
    utilizations_list = []
    
    # Lists to store results for plotting
    node1_results_first = []
    node2_results_first = []
    node1_results_second = []
    node2_results_second = []
    
    for idx, mu in enumerate(service_rates):
        print(f"\nRunning simulation with service rate μ = {mu}")
        nodes = {
            1: Queue(1, 'exponential', {'mean': 1/mu}),
            2: Queue(2, 'exponential', {'mean': 1/mu})
        }

        results = simulate_queue_network(
            nodes=nodes,
            routing_matrix=routing_matrix,
            external_arrival_rates=external_arrival_rates,
            arrival_distributions=arrival_distributions,
            arrival_params=arrival_params,
            total_jobs=total_jobs,
            warmup_period=warmup_period,
            simulation_period=simulation_period,
            time_interval=time_interval,
            seed=seed
        )

        mean_sojourn_time = results['mean_sojourn_time']
        confidence_interval = results['confidence_interval']
        mean_sojourn_times.append(mean_sojourn_time)
        confidence_intervals.append(confidence_interval)
        utilizations = results['utilizations']
        utilizations_list.append(utilizations)
        all_results[mu] = results

        print(f"Mean Sojourn Time: {mean_sojourn_time:.4f} ± {confidence_interval:.4f} seconds")
        for node_id, utilization in utilizations.items():
            print(f"Utilization of Node {node_id}: {utilization:.4f}")
        print("-" * 40)

        # Collect data for plotting
        if idx < 2:
            # First two runs (μ = 4.0 and 5.0)
            for node_id in [1, 2]:
                stats = results['node_stats'][node_id]
                times = stats['times']
                queue_lengths = stats['queue_lengths']
                if node_id == 1:
                    node1_results_first.append((mu, times, queue_lengths))
                else:
                    node2_results_first.append((mu, times, queue_lengths))
        else:
            # Remaining runs (μ = 6.0 to 10.0)
            for node_id in [1, 2]:
                stats = results['node_stats'][node_id]
                times = stats['times']
                queue_lengths = stats['queue_lengths']
                if node_id == 1:
                    node1_results_second.append((mu, times, queue_lengths))
                else:
                    node2_results_second.append((mu, times, queue_lengths))

    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Subplot 1: Node 1, first two runs
    for mu, times, queue_lengths in node1_results_first:
        axs[0, 0].plot(times, queue_lengths, label=f'μ = {mu}')
    axs[0, 0].set_title('Node 1 Queue Lengths (μ = 4.0 and 5.0)')
    axs[0, 0].set_xlabel('Time (seconds)')
    axs[0, 0].set_ylabel('Queue Length')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Subplot 2: Node 2, first two runs
    for mu, times, queue_lengths in node2_results_first:
        axs[0, 1].plot(times, queue_lengths, label=f'μ = {mu}')
    axs[0, 1].set_title('Node 2 Queue Lengths (μ = 4.0 and 5.0)')
    axs[0, 1].set_xlabel('Time (seconds)')
    axs[0, 1].set_ylabel('Queue Length')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Subplot 3: Node 1, remaining runs
    for mu, times, queue_lengths in node1_results_second:
        axs[1, 0].plot(times, queue_lengths, label=f'μ = {mu}')
    axs[1, 0].set_title('Node 1 Queue Lengths (μ = 6.0 to 10.0)')
    axs[1, 0].set_xlabel('Time (seconds)')
    axs[1, 0].set_ylabel('Queue Length')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Subplot 4: Node 2, remaining runs
    for mu, times, queue_lengths in node2_results_second:
        axs[1, 1].plot(times, queue_lengths, label=f'μ = {mu}')
    axs[1, 1].set_title('Node 2 Queue Lengths (μ = 6.0 to 10.0)')
    axs[1, 1].set_xlabel('Time (seconds)')
    axs[1, 1].set_ylabel('Queue Length')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

    # Plot mean sojourn times vs. service rates with confidence intervals
    plt.figure()
    plt.errorbar(service_rates, mean_sojourn_times, yerr=confidence_intervals, fmt='o-', capsize=5)
    plt.xlabel('Service Rate (μ)')
    plt.ylabel('Mean Sojourn Time')
    plt.title('Mean Sojourn Time vs. Service Rate with 95% Confidence Intervals')
    plt.grid(True)
    plt.show()

    return all_results



# Main 
def main():
    # network topology mu1 = mu2  (mean service time = 1/mu )
    num_nodes = 2
    service_rates_system = [8.0, 8.0]
    nodes = {
        1: Queue(1, 'exponential', {'mean': 1/service_rates_system[0]}),  
        2: Queue(2, 'exponential', {'mean': 1/service_rates_system[1]})   
    }

    partitioned_nodes = {
        1: {
            'event_stack': EventStack(1),
            'nodes': {
                1: Queue(1, 'exponential', {'mean': 1/service_rates_system[0]}),
            }
        },
        2: {
            'event_stack': EventStack(2),
            'nodes': {
                2: Queue(2, 'exponential', {'mean': 1/service_rates_system[1]}),
            }
        }
    }
    # Routing matrix: Node 1 -> Node 2 -> Exit
    #{ from_node: {to_node: probability} }
    routing_matrix = {
        1: {2: 1.0}, 
        2: {}         
    }
    # for more general queueing network, external arrival rate for each node {node_num: lambda}
    external_arrival_rates = {
        1: 5.0  #  lambda
    }
    arrival_distributions = {
        1: 'exponential'
    }
    arrival_params = {
        1: {'mean': 1 / external_arrival_rates[1]}  
    }

    total_jobs = 10000
    warmup_period = 10
    simulation_period = 1900
    seed = 12981  
    batch_size = 20
    service_rates = [4.0 ,5.0, 6.0, 7.0 , 8.0, 9.0, 10.0]  
    mean_sojourn_times = []
    utilizations_list = []

    
    results = simulate_partition_queue_network(
        partitions=partitioned_nodes,
        routing_matrix=routing_matrix,
        external_arrival_rates=external_arrival_rates,
        arrival_distributions=arrival_distributions,
        arrival_params=arrival_params,
        total_jobs=total_jobs,
        warmup_period=warmup_period,
        simulation_period=simulation_period,
        time_interval=1800,
        seed=seed
    )
    all_results = run_simulations_and_plot(service_rates, routing_matrix, external_arrival_rates, arrival_distributions, arrival_params, total_jobs, warmup_period, simulation_period, 1800, seed)  
    #results = all_results[8.0]  # Results for service rate μ = 8.0
 

    mean_sojourn_time = results['mean_sojourn_time']
    confidence_interval = results['confidence_interval']
    utilizations = results['utilizations']
    simulation_time = results['simulation_time']
    sojourn_times = results['sojourn_times']

    
    plot_batch_means(sojourn_times, batch_size=batch_size)
    
    fig, axs = plt.subplots(len(results['node_stats']), 1, figsize=(10, 6*len(results['node_stats'])))
    if len(results['node_stats']) == 1:
        axs = [axs]  # Ensure axs is iterable
    for i, (node_id, stats) in enumerate(results['node_stats'].items()):
        times = stats['times']
        queue_lengths = stats['queue_lengths']
        if times and queue_lengths:
            axs[i].step(times, queue_lengths, where='post')
            axs[i].set_xlabel('Time (hours)')
            axs[i].set_ylabel(f'Queue Length at Node {node_id}')
            axs[i].set_title(f'Queue Length Over Time at Node {node_id}')
            axs[i].grid(True)

    plt.tight_layout()
    plt.show()

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
    

    # Plot mean sojourn time with confidence interval
    plt.figure(figsize=(6, 6))
    plt.bar(['Mean Sojourn Time'], [mean_sojourn_time], yerr=[confidence_interval], capsize=10, color='skyblue')
    plt.ylabel('Time (seconds)')
    plt.title('Mean Sojourn Time with 95% Confidence Interval')
    plt.show()


    # Validation with Jackson's Theorem AND LITTLE 's L
    validate_with_jackson(simulation=results,arrival_rate=external_arrival_rates[1],service_rates=service_rates_system, num_queues=num_nodes)
    calculate_little_l(results, external_arrival_rates, mean_sojourn_time)




if __name__ == "__main__":
    main()
