import matplotlib.pyplot as plt
import os

def mm1_queue_calculator(lambda_rate, mu_rate, drop_out_rate):
    p = lambda_rate / mu_rate

    E_r = (1 / mu_rate) / (1 - p)
    E_w = E_r - (1 / mu_rate)
    E_n = p / (1 - p)
    E_nq = pow(p,2) / (1-p)


    return {
        'Gateway Utilization (p)': p,
        'Mean # of customers in the system (E[n])': E_n,
        'Response Time (E[r])': E_r,
        'Mean Waiting Time (E[w])': E_w,
        'Mean # of customers in the queue (E[nq])': E_nq
    }

unit_of_time = 3000

service_rate_router = 1/5.488    # Service rate (customers per minute)
drop_out_rate = 0.       # Dropout rate

thetas = list(range(6, 61, 2))
E_n_values_theory = []
E_nq_values_theory = []
E_w_values_theory = []
E_r_values_theory = []

p_values = []

for theta in thetas:
    lambda_rate = 1/theta
    results = mm1_queue_calculator(lambda_rate, service_rate_router, drop_out_rate)
    p_values.append(results['Gateway Utilization (p)'])
    E_n_values_theory.append(results['Mean # of customers in the system (E[n])'])
    E_nq_values_theory.append(results['Mean # of customers in the queue (E[nq])'])
    E_w_values_theory.append(results['Mean Waiting Time (E[w])'])
    E_r_values_theory.append(results['Response Time (E[r])'])



plt.figure(figsize=(10, 5))
plt.plot(thetas, p_values, marker='o', color='orange')
plt.xlabel("Inter-arrival time (minutes)")
plt.ylabel("Resource Utilization of system")
plt.title("System Utilization")
plt.grid(True)
plt.legend()
plt.savefig("analysis/utilization.png", dpi=300)
plt.close()


plt.figure(figsize=(10, 5))
plt.plot(thetas, E_n_values_theory, marker='.', color='red', label="Number of jobs in the system")
plt.plot(thetas, E_nq_values_theory, marker='.', color='orange', label="Number of jobs waiting for service")
plt.xlabel("Inter-arrival time (minutes)")
plt.ylabel("Number of jobs")
plt.title("Number of jobs in the system")
plt.grid(True)
plt.legend()
plt.savefig("analysis/en_and_enq_of_router_tc1.png", dpi=300)  # Save with 300 DPI for high quality
plt.close()


plt.figure(figsize=(10, 5))
plt.plot(thetas, E_w_values_theory, marker='.', color='red', label="Waiting time")
plt.plot(thetas, E_r_values_theory, marker='.', color='orange', label="Response time")
plt.xlabel("Inter-arrival time (minutes)")
plt.ylabel("Time unit (minutes)")
plt.title("Time analysis")
plt.grid(True)
plt.legend()
plt.savefig("analysis/ew_and_er_of_router_tc1.png", dpi=300)
plt.close()

