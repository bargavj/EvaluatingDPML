import math
import numpy as np
from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
from scipy.stats import norm

# optimal sigma values for GDP mechanism for batch size = 500, epochs = 50, training size = 50000, delta = 1e-5.
gdp_noise_multiplier = {0.1:22, 1:2.75, 10:0.675, 100:0.428}


# Parameters can be varied to find optimal epsilon values for different settings
delta = 1e-5
epochs = 50
n = 50000
batch_size = 500
sigma = 22

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
        eps2, dlt = get_gdp_privacy_spent(orders, mu, target_delta=delta)
        if dlt > delta:
            print("Error: Could not find suitable privacy parameters!")
        print('For delta= %f' % dlt, ',the epsilon is: %.2f' % eps2)