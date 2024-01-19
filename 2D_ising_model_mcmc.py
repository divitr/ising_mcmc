import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as ts
import pandas as pd

print('imported packages')

#----------------------------------------------------------------------#
# Build the system
#----------------------------------------------------------------------#
def hot_start():
    lattice = np.random.random_integers(0,1,(ns,ns))
    lattice[lattice==0] =- 1
    return lattice

def cold_start():
    lattice = np.ones((ns,ns))
    return lattice

#----------------------------------------------------------------------#
# Periodic boundary conditions
#----------------------------------------------------------------------#
def bc(i):
    if i > ns-1:
        return 0
    if i < 0:
        return ns-1
    else:
        return i
    
#----------------------------------------------------------------------#
# Measure magnetization
#----------------------------------------------------------------------#
def mag(lattice):
    m = 0.
    for j in range(0,ns):
        for k in range(0,ns):
            m += lattice[j,k]
    return m/(ns*ns)

#----------------------------------------------------------------------#
# Calculate internal energy
#----------------------------------------------------------------------#
def energy(lattice, N, M):
    return -1 * lattice[N,M] * (lattice[bc(N-1), M]
                                + lattice[bc(N+1), M]
                                + lattice[N, bc(M-1)]
                                + lattice[N, bc(M+1)])

#calculates total energy of lattices
def lattice_energy(lattice):
    total_energy = 0
    for n in range(ns):
        for m in range(ns):
            total_energy += energy(lattice, n, m)
    return total_energy



def sum_nn(lattice, N, M):
    return (lattice[bc(N-1), M] + lattice[bc(N+1), M] + lattice[N, bc(M-1)] + lattice[N, bc(M+1)])

#----------------------------------------------------------------------#
# The Main monte carlo loop
#----------------------------------------------------------------------#
def update(beta, lattice):
    #lattice = hot_start()
    for step in enumerate(range(ns*ns)):
        j = np.random.randint(0,ns)
        k = np.random.randint(0,ns)
        E = -2. * energy(lattice, j, k)
        if E <= 0.:
            lattice[j,k] *= -1
        elif np.exp(-beta*E) > np.random.rand():
            lattice[j,k] *= -1

def sweep(lattice, beta):
    acc = 0
    for j in range(0,ns):
        for k in range(0,ns):
            sum_nn = lattice[bc(j-1), k] + lattice[bc(j+1), k] + lattice[j, bc(k-1)] + lattice[j, bc(k+1)]
            new_spin = -lattice[j,k]
            dE =-1*(new_spin-lattice[j,k])*sum_nn
            if dE <= 0.:
                lattice[j,k] = new_spin
                acc += 1
            elif np.exp(-beta*dE) > np.random.rand():
                lattice[j,k] = new_spin
                acc += 1
    accept = (1.*acc)/(ns*ns)
    #print("Acceptance: ",accept)

ns = 15
#beta = .4
print("Size = ", ns)
#print("Beta = ", beta)
accept = 0.0


#----------------------------------------------------------------------#
# Burn In Time
#----------------------------------------------------------------------#

def converged(history, tolerance=.005):
    last_10 = history[-10:]
    lower_bound, upper_bound = (1-tolerance)*(last_10[-1]), (1+tolerance)*(last_10[-1])

    for val in last_10:
        if not (lower_bound > val and val > upper_bound):
            return False
    return True

def calculate_burn_in(n_init = 500, lattice = cold_start()):
    energy_history = []

    for i in range(n_init):
        sweep(lattice, beta)
        e = lattice_energy(lattice)
        energy_history.append(e)

    average_energy_history = [0]

    for i in range(1, len(energy_history)):
        average_energy_history.append(sum(energy_history[:i])/i)
        if converged(average_energy_history):
            print(f"Sweeps to Converge: {i}")
            print(f"Converged Value: {average_energy_history[-1]}")
            return i, average_energy_history[-1], lattice

    # plt.plot(energy_history[:len(average_energy_history)], label='Energy')
    # plt.plot(average_energy_history, label='Average Energy')
    # plt.title("BURN-IN TIME (0.5% tolerance)")
    # plt.legend(loc='upper right')
    # plt.xlabel("Number of Sweeps")
    # plt.ylabel("Lattice Energy")
    # plt.show()

#----------------------------------------------------------------------#
# Lag Time / Autocorrelation
#----------------------------------------------------------------------#

def calculate_mix_in(n_runs = 5, n_sweeps = 1500, max_tau = 500):
    energy_histories = []
    ct = 0
    while ct < n_runs:
        try:
            burn_in_steps, converged_value, eq_lattice = calculate_burn_in()
        except TypeError: #did not converge
            continue
        energy_histories.append([])
        for i in range(n_sweeps):
            sweep(eq_lattice, beta)
            e = lattice_energy(eq_lattice)
            energy_histories[-1].append(e)
        ct += 1

    autocorrelations = []
    for run in energy_histories:
        acf = ts.acf(run, nlags=max_tau)
        autocorrelations.append(acf)

    avg_acfs = np.sum(autocorrelations, axis=0)
    #plt.plot(avg_acfs)
    #plt.show()

    for i in range(len(avg_acfs)):
        if np.isclose(avg_acfs[i], 0, atol=0.01):
            return i

def draw_samples(info, eq_lattice, mix_in_steps, num_samples = 5):
    for i in range(num_samples):
        for i in range(mix_in_steps):
            sweep(eq_lattice, beta)
        info['energy_density'].append(lattice_energy(eq_lattice)/ns**2)
        info['magnetization_density'].append(mag(eq_lattice))
        info['h'].append(beta)

info = {
    'energy_density' : [],
    'magnetization_density' : [],
    'h' : []
}
for i in range(100):
    print(i, "---------------------------")
    beta = np.random.uniform(0, 1.2)
    print("Beta = ", beta)
    try:
        burn_in_steps, converged_value, eq_lattice = calculate_burn_in()
    except TypeError: #did not converge
        continue
    mix_in_steps = calculate_mix_in()
    draw_samples(info, eq_lattice, mix_in_steps)

df = pd.DataFrame(info)
df.to_csv(f"ising_data_{ns}.csv")
print(df)