import math
import time
import numpy as np

from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
from scipy.stats import norm

MAX_SIGMA = 1e6
EPS_TOLERANCE = 8e-3 # lower tolerance may not converge


def compute_gdp_mu(sampling_rate, sigma, steps):
    """
    Calculates the mu for the GDP mechanism for a given sigma value.
    """
    return sampling_rate * np.sqrt(steps * (np.exp(1 / sigma**2) - 1) )
 

def get_gdp_privacy_spent(mu, target_delta):
    """
    Returns the estimated optimal epsilon and delta values for the mu-GDP 
    mechanism given a threshold on delta.
    """
    eps_orders = np.logspace(-3.0, 2.0, num=10000)
    low, high = 0, len(eps_orders) - 1
    delta = 1
    while low <= high:
        mid = (low + high) // 2
        eps = eps_orders[mid]
        dlt = norm.cdf(-eps / mu + mu / 2, loc=0, scale=1) - np.exp(eps) * norm.cdf(-eps / mu - mu / 2, loc=0, scale=1)
        if dlt <= target_delta:
            delta = dlt
            high = mid - 1
        else:
            low = mid + 1
    return eps, delta


class accountant:
    """
    Privacy accountant class that calculates the noise multiplier (sigma) for 
    given privacy parameters (epsilon and delta), and the accounting type.
    """
    
    def __init__(self, data_size, batch_size, epochs, target_delta, dp_type):
        """
        data_size: int -- size of the model training set
        batch_size: int -- batch size used in model training
        epochs: int -- number of training iterations over the training set
        target_delta : float -- tolerance on failure probability for privacy accounting
        dp_type: str -- type of differential privacy accounting used: 'dp', 'adv_cmp', 'zcdp', 'rdp', 'gdp'
        """
        self.data_size = data_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.target_delta = target_delta
        self.dp_type = dp_type
        
        self.steps_per_epoch = self.data_size // self.batch_size
        self.steps = self.epochs * self.steps_per_epoch
        self.sampling_rate = self.batch_size / self.data_size
    
    
    def get_noise_multiplier(self, target_epsilon):
        """
        Return the noise multiplier (sigma) for the given privacy accounting mechanism.
        """
        if self.dp_type == 'dp':
            return self.epochs * np.sqrt(2 * np.log(1.25 * self.epochs / self.target_delta)) / target_epsilon
        
        elif self.dp_type == 'adv_cmp':
            return np.sqrt(self.epochs * np.log(2.5 * self.epochs / self.target_delta)) * (np.sqrt(np.log(2 / self.target_delta) + 2 * target_epsilon) + np.sqrt(np.log(2 / self.target_delta))) / target_epsilon
        
        elif self.dp_type == 'zcdp':
            return np.sqrt(self.epochs / 2) * (np.sqrt(np.log(1 / self.target_delta) + target_epsilon) + np.sqrt(np.log(1 / self.target_delta))) / target_epsilon
        
        else: # if self.dp_type == 'rdp' or 'gdp'
            return self.search_optimal_noise_multiplier(target_epsilon)
    
    
    def search_optimal_noise_multiplier(self, target_epsilon):
        """
        Performs binary search to get the optimal value for noise multiplier (sigma) for RDP and GDP accounting mechanisms. Functionality adapted from Opacus (https://github.com/pytorch/opacus).
        """
        eps_high = float("inf")
        sigma_low, sigma_high = 0, 10
        orders = [1 + x / 100.0 for x in range(1, 1000)] + list(range(12, 1200))
        
        while eps_high > target_epsilon:
            sigma_high = 2 * sigma_high
            
            if self.dp_type == 'rdp':
                rdp = compute_rdp(self.sampling_rate, sigma_high, self.steps, orders)
                eps_high, _, _ = get_privacy_spent(orders, rdp, target_delta=self.target_delta)
            else: # if self.dp_type == 'gdp'
                mu = compute_gdp_mu(self.sampling_rate, sigma_high, self.steps)
                eps_high, delta = get_gdp_privacy_spent(mu, target_delta=self.target_delta)
                if delta > self.target_delta:
                    raise ValueError("Could not find suitable privacy parameters.")
            
            if sigma_high > MAX_SIGMA:
                raise ValueError("The privacy budget is too low.")

        while target_epsilon - eps_high > EPS_TOLERANCE * target_epsilon:
            sigma = (sigma_low + sigma_high) / 2
            
            if self.dp_type == 'rdp':
                rdp = compute_rdp(self.sampling_rate, sigma, self.steps, orders)
                eps, _, _ = get_privacy_spent(orders, rdp, target_delta=self.target_delta)
            else: # if self.dp_type == 'gdp'
                mu = compute_gdp_mu(self.sampling_rate, sigma, self.steps)
                eps, delta = get_gdp_privacy_spent(mu, target_delta=self.target_delta)
            
            if eps < target_epsilon:
                sigma_high = sigma
                eps_high = eps
            else:
                sigma_low = sigma
        
        return sigma_high
    
    
if __name__ == '__main__':
    ac = accountant(
        data_size=50000, 
        batch_size=500, 
        epochs=50, 
        target_delta=1e-5, 
        dp_type='gdp')
    t0 = time.time()
    print("Sigma = %.4f" % ac.get_noise_multiplier(target_epsilon=10))
    print("%.2f seconds to execute" % (time.time() - t0))