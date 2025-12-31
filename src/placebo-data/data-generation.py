
import numpy as np


class DataGenerator:
    def __init__(self, num_samples: int, num_features: int, seed: int = 123):
        self.num_samples = num_samples
        self.num_features = num_features
        self.seed = seed

    def generate_data(self):
        seed = np.random.BitGenerator(self.seed)
        data = np.random.Generator(seed).random((self.num_samples, self.num_features))
        labels = np.random.Generator(seed).integers(2, size=self.num_samples)
        return data, labels

    def generate_data_with_noise(self, noise_level: float):
        seed = np.random.BitGenerator(self.seed)
        data = np.random.Generator(seed).random((self.num_samples, self.num_features))
        labels = np.random.Generator(seed).integers(2, size=self.num_samples)
        noisy_data = data + noise_level * np.random.Generator(seed).normal(size=data.shape)
        return noisy_data, labels

# TODO: Finish this data generation method regarding to article.
# TODO: Stress ridge and ridgeless regression with this created data generation class with several parameters count.
# TODO: Create a notebook to visualize the double descent phenomenon.
