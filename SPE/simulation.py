import logging
import simpy
import random
import os


def configure_logger(log_path):
    """Configures the logger with a specific file path."""
    logger = logging.getLogger('my_logger')
    logger.handlers = []  # Clear existing handlers to avoid duplicate logs
    logger.setLevel(logging.INFO)

    # Create handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_path)

    # Set level for handlers
    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


class numberCustomer:
    def __init__(self):
        self.num_customer_in_service_queue       = 0


class Customer:
    def __init__(self, 
                 env: simpy.Environment, 
                 customer_id: str,
                 service_rate: float,
                 num_customer: numberCustomer):
        self.env = env
        self.customerID = customer_id
        self.num_customer = num_customer
        self.service_rate = service_rate
        self.agent = simpy.Resource(env, capacity=1)

    def serve_customer(self):
        service_time = random.expovariate(self.service_rate)
        logger.info(f"[Start] Time {self.env.now}: {self.customerID} starts service")
        yield self.env.timeout(service_time)
        logger.info(f"[Complete] Time {self.env.now}: {self.customerID} has completed service")
        

    def run(self):
        with self.agent.request() as request:
            yield request
            logger.info(f"[Dequeue] Time {self.env.now}: {self.customerID} dequeued")

            yield self.env.process(self.serve_customer())
            self.num_customer.num_customer_in_service_queue -= 1

        logger.info(f"[Leave system] Time {self.env.now}: {self.customerID} leaves the system")


# ArrivalEvent Class
class Generator:
    def __init__(self, 
                 env: simpy.Environment, 
                 arrival_rate: float,
                 service_rate: float, 
                 num_customer: numberCustomer):
        self.env = env
        self.customer_id = 0  # To assign unique IDs to customers

        self.arrival_rate = arrival_rate
        self.service_rate = service_rate

        self.env.process(self.generateArrival())

        self.num_customer = num_customer

    def generateArrival(self):
        while True:
            self.customer_id += 1
            customer = Customer(self.env, f'Customer_{self.customer_id}', self.service_rate, self.num_customer)

            logger.info(f"[Arrival] Time {self.env.now}: {customer.customerID} arrives")
            logger.info(f"[Enqueue] Time {self.env.now}: Customer_{self.customer_id} enqueued")

            logger.info(f"[Update] Time {self.env.now}: Number of customer(s) waiting are {self.num_customer.num_customer_in_service_queue}")
            self.num_customer.num_customer_in_service_queue  += 1
            self.env.process(customer.run())

            inter_arrival_time = random.expovariate(self.arrival_rate)
            yield self.env.timeout(inter_arrival_time)


# Main simulation function
def main(arrival_rate):
    global logger
    log_path = f'log/local_{arrival_rate}.log'
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open('log/local_{}.log'.format(arrival_rate),'w') as file:
        pass

    logger = configure_logger(log_path)


    random.seed(42)
    env = simpy.Environment()

    service_rate = 1/5.488

    # Create number of customer in router
    num_customer = numberCustomer()
    # Create ArrivalEvent
    Generator(env, arrival_rate, service_rate, num_customer)

    env.run(until=3000)
    

if __name__ == "__main__":
    import numpy as np
    # from 1 min to 1 day
    avg_arrival = np.arange(1, 61, 2)
    arrival_rates = 1/avg_arrival
    print(arrival_rates, len(arrival_rates))
    breakpoint()
    for arrival_rate in arrival_rates:
        main(arrival_rate=arrival_rate)