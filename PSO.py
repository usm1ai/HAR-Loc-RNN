import numpy as np
from scipy.stats import zscore

def hybridize_features(batch):
    batch_size, n_features = batch.shape
    hybridized = np.zeros_like(batch)
    for i in range(batch_size):
        for j in range(n_features):
            hybridized[i, j] = batch[np.random.randint(0, batch_size), j]
    return hybridized

def pso_feature_optimizer(X, num_iterations=10, batch_size=10, random_state=42):
    np.random.seed(random_state)
    n_samples, n_features = X.shape
    num_batches = n_samples // batch_size
    global_best_features = []
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        batch = X[start_idx:end_idx]
        particles = hybridize_features(batch)
        num_particles = particles.shape[0]
        positions = particles.copy()
        velocities = np.random.uniform(-1, 1, size=positions.shape)
        local_best_scores = np.full(num_particles, -np.inf)
        local_best_positions = np.zeros_like(positions)
        global_best_score = -np.inf
        global_best_position = None
        for iteration in range(num_iterations):
            for i in range(num_particles):
                z_scores = zscore(positions[i])
                fitness_score = np.abs(z_scores).sum()
                if fitness_score > local_best_scores[i]:
                    local_best_scores[i] = fitness_score
                    local_best_positions[i] = positions[i]
                if fitness_score > global_best_score:
                    global_best_score = fitness_score
                    global_best_position = positions[i]
            w = 0.5
            c1 = 2.0
            c2 = 2.0
            for i in range(num_particles):
                r1, r2 = np.random.rand(), np.random.rand()
                cognitive = c1 * r1 * (local_best_positions[i] - positions[i])
                social = c2 * r2 * (global_best_position - positions[i])
                velocities[i] = w * velocities[i] + cognitive + social
                positions[i] += velocities[i]
        global_best_features.append(global_best_position)
    return global_best_features



