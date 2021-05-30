import math
import numpy as np
from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
from scipy.stats import norm

# optimal sigma values for GDP mechanism for the batch size = 200, epochs = 100, training set size = 10000, delta = 1e-5.
gdp_noise_multiplier = {0.01:350, 0.05:82, 0.1:44, 0.5:10, 1:5.4, 5:1.43, 10:0.955, 50:0.564, 100:0.498}
# optimal sigma values for GDP mechanism for the batch size = 200, epochs = 30, training set size = 10000, delta = 1e-5.
gdp_noise_multiplier = {0.01:190, 0.05:45, 0.1:24, 0.5:5.5, 1:3, 5:0.94, 10:0.701, 50:0.481, 100:0.438}


# Parameters can be varied to find optimal epsilon values for different settings
delta = 1e-5
epochs = 30
n = 10000
batch_size = 200
sigma = 0.94

steps_per_epoch = n // batch_size
T = epochs * steps_per_epoch
p = batch_size / n
print(sigma, p, T)


def compute_gdp_mu(p, sigma, T):
    return p * np.sqrt(T * (np.exp(1 / sigma**2) - 1) )

def get_gdp_privacy_spent(eps_orders, mu, target_delta):
    for eps in eps_orders:
	    dlt = norm.cdf(-eps / mu + mu / 2, loc=0, scale=1) - np.exp(eps) * norm.cdf(-eps / mu - mu / 2, loc=0, scale=1)
	    if dlt <= target_delta:
		    break
    return eps, dlt

if __name__ == '__main__':
    eps2 = 0.01
    for step in range(T, T+1):
        print('\nStep: %d' % step)
        print('-'*10 + 'RDP' + '-'*10 + '\n')
        orders = [1 + x / 100.0 for x in range(1, 1000)] + list(range(12, 1200))
        rdp = compute_rdp(p, sigma, step, orders)
        eps, _, opt_order = get_privacy_spent(orders, rdp, target_delta=delta)
        print('For delta= %f' % delta, ',the epsilon is: %.2f' % eps)
        
        print('-'*10 + 'GDP' + '-'*10 + '\n')
        mu = compute_gdp_mu(p, sigma, step)
        print(mu)
        eps_orders = np.arange(eps2, 100, 0.01)
        eps2, dlt = get_gdp_privacy_spent(eps_orders, mu, target_delta=delta)
        if dlt > delta:
            print("Error: Could not find suitable privacy parameters!")
        print('For delta= %f' % delta, ',the epsilon is: %.2f' % eps2)