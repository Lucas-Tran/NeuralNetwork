import json

import numpy as np


def parse_data(file_name):
    with open(file_name) as f:
        data = json.load(f)
        house_sizes = []
        num_bedrooms = []
        num_bathrooms = []
        house_ages = []
        estimated_prices = []
        M = 0
        for training_example in data:
            house_sizes.append(training_example['house_size'])
            num_bedrooms.append(training_example['num_bedrooms'])
            num_bathrooms.append(training_example['num_bathrooms'])
            house_ages.append(training_example['house_age'])
            estimated_prices.append(training_example['estimated_price'])
            M += 1
    return M, np.array([house_sizes, num_bedrooms, num_bathrooms, house_ages]), np.array([estimated_prices])