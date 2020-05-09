# To avoid numerical inconsistency in calculating log
SMALL_VALUE = 1e-6

# Seed for random number generator
SEED = 21312

# Optimal sigma values for RDP mechanism with varying epsilon values for the batch size = 200, training set size = 10000, delta = 1e-5; indexed by epochs
# Texas-100 uses epochs = 30
# Purchase-100 uses epochs = 100
rdp_noise_multiplier = {
    30: {0.01: 290, 0.05: 70, 0.1: 36, 0.5: 7.6, 1: 3.9, 5: 1.1, 10: 0.79, 50: 0.445, 100: 0.356, 500: 0.206, 1000: 0.157},
    100: {0.01: 525, 0.05: 150, 0.1: 70, 0.5: 13.8, 1: 7, 5: 1.669, 10: 1.056, 50: 0.551, 100: 0.445, 500: 0.275, 1000: 0.219}
}

# Optimal sigma values for GDP mechanism with varying epsilon values for the batch size = 200, training set size = 10000, delta = 1e-5; indexed by epochs
# Texas-100 uses epochs = 30
# Purchase-100 uses epochs = 100
gdp_noise_multiplier = {
    30: {0.01: 190, 0.05: 45, 0.1: 24, 0.5: 5.5, 1: 3, 5: 0.94, 10: 0.701, 50: 0.481, 100: 0.438},
    100: {0.01: 350, 0.05: 82, 0.1: 44, 0.5: 10, 1: 5.4, 5: 1.43, 10: 0.955, 50: 0.564, 100: 0.498}
}