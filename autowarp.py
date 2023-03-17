import torch
import torch.optim as optim
import numpy as np
from autoencoders import LinearAutoencoder, ConvAutoencoder, ConvLinearAutoEncoder, \
    get_distance_matrix

class AutoWarp:

    def __init__(self, model, data, latent_size, p=0.5, max_iterations=100, batch_size=25, lr=0.1):
        self.model = model
        self.data = data
        self.latent_size = latent_size
        self.p = p*100
        self.max_iterations = max_iterations
        self.batch_size = batch_size
        self.lr = lr

        self.alpha = torch.rand(1, requires_grad=True)
        self.gamma = torch.rand(1, requires_grad=True)
        self.epsilon = torch.rand(1, requires_grad=True)

        self.optimizer = optim.SGD([self.alpha, self.gamma, self.epsilon], lr=self.lr)

    def encodings(self):

        with torch.no_grad():
            if type(self.model) == LinearAutoencoder:
                encodings = self.model.encoder(torch.tensor(self.data).float().permute(0, 2, 1))
            else:
                encodings = self.model.encoder(torch.tensor(self.data).float())
        return encodings

    def euclidian_distance(self):

        return get_distance_matrix(self.model, self.data, self.latent_size, distance_metric="euclidean")

    def sample_trajectory_pairs(self, euclidian_distance):

        # Compute delta
        delta = np.percentile(euclidian_distance, self.p)

        # Create a mask for pairs with distance in the latent space less than delta
        close_pairs_mask = np.triu(euclidian_distance < delta, k=1)

        # Get the indices of the close pairs and all pairs
        close_pairs_indices = np.column_stack(np.where(close_pairs_mask))
        all_pairs_indices = np.column_stack(np.where(np.triu(np.ones_like(euclidian_distance), k=1)))

        # Randomly sample S pairs of trajectories from the close pairs and all pairs
        close_pairs = close_pairs_indices[np.random.choice(close_pairs_indices.shape[0], self.batch_size, replace=False)]
        all_pairs = all_pairs_indices[np.random.choice(all_pairs_indices.shape[0], self.batch_size, replace=False)]

        return close_pairs, all_pairs

    @staticmethod
    def warping_distance(t_A, t_B, alpha, gamma, epsilon):

        n, m = len(t_A), len(t_B)
        cost_matrix = torch.zeros((n + 1, m + 1), requires_grad=False)

        for i in range(1, n + 1):
            cost_matrix[i, 0] = float('inf')
        for j in range(1, m + 1):
            cost_matrix[0, j] = float('inf')

        pairwise_euc = torch.cdist(t_A.unsqueeze(0), t_B.unsqueeze(0), p=2).squeeze()
        sigma = lambda x, y: y * torch.tanh(x / y)

        # Calculate the cost matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                c1 = sigma(pairwise_euc[i - 1, j - 1], epsilon / (1 - epsilon))
                c2 = alpha / (1 - alpha) * sigma(pairwise_euc[i - 1, j - 1], epsilon / (1 - epsilon)) + gamma
                updated_cost = c1 + cost_matrix[i - 1, j - 1]

                if i > 1:
                    updated_cost = torch.min(updated_cost, c2 + cost_matrix[i - 1, j])
                if j > 1:
                    updated_cost = torch.min(updated_cost, c2 + cost_matrix[i, j - 1])

                cost_matrix[i, j] = updated_cost

        return cost_matrix[-1, -1]

    def compute_gradients_and_beta_hat_torch(self, encodings, P_c, P_all):

        beta_hat_numerator = torch.tensor(0.0)
        beta_hat_denominator = torch.tensor(0.0)

        # calculate beta_hat
        for t_i, t_j in P_c:
            beta_hat_numerator += self.warping_distance(encodings[t_i], encodings[t_j], self.alpha, self.gamma, self.epsilon)

        for t_i, t_j in P_all:
            beta_hat_denominator += self.warping_distance(encodings[t_i], encodings[t_j], self.alpha, self.gamma, self.epsilon)

        beta_hat = beta_hat_numerator.detach() / beta_hat_denominator.detach()
        beta_hat.requires_grad = True
        self.optimizer.zero_grad()
        beta_hat.backward()
        self.optimizer.step()

        return self.alpha, self.gamma, self.epsilon, beta_hat

    def learn_metric(self):

        iteration = 0
        convergence = False

        # Setup calculations
        encodings = self.encodings()
        euclidian_distance = self.euclidian_distance()

        while not convergence and iteration < self.max_iterations:

            # Sample S pairs of trajectories from the euclidian distance matrix
            close_pairs, all_pairs = self.sample_trajectory_pairs(euclidian_distance)

            # Compute gradients and beta_hat
            if iteration>0:
                beta_hat_old = beta_hat
            beta_hat, self.alpha, self.gamma, self.epsilon = self.compute_gradients_and_beta_hat_torch(encodings, close_pairs, all_pairs)

            iteration += 1

            # Check for convergence in beta_hat
            if iteration > 1:
                if abs(beta_hat - beta_hat_old) < 0.0001:
                    convergence = True

        print("alpha: ", self.alpha.item())
        print("gamma: ", self.gamma.item())
        print("epsilon: ", self.epsilon.item())
        print("betaCV: ", beta_hat.item())

    def create_distance_matrix(self):
        encodings = self.encodings()
        num_trajectories = encodings.shape[0]
        distance_matrix = torch.zeros((num_trajectories, num_trajectories), dtype=torch.float)

        for i in range(num_trajectories):
            for j in range(i + 1, num_trajectories):
                distance = self.warping_distance(encodings[i], encodings[j], self.alpha, self.gamma, self.epsilon)
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

        return distance_matrix
