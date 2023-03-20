import torch
import torch.optim as optim
import numpy as np
from models.autoencoders import LinearAutoencoder, get_distance_matrix
from numba import jit
import concurrent.futures


class AutoWarp:

    def __init__(self, model, data, latent_size, p=0.5, max_iterations=100, batch_size=25, lr=0.1):
        self.model = model
        self.data = data
        self.latent_size = latent_size
        self.p = p * 100
        self.max_iterations = max_iterations
        self.batch_size = batch_size
        self.lr = lr

        self.alpha = torch.rand(1, requires_grad=True)
        self.gamma = torch.rand(1, requires_grad=True)
        self.epsilon = torch.rand(1, requires_grad=True)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.optimizer = optim.Adam([self.alpha, self.gamma, self.epsilon], lr=self.lr)

    def encodings(self):
        """
        Calculates latent representation of the data from the encoder of the model
        :return: Latent representation of the data
        """

        with torch.no_grad():
            if type(self.model) == LinearAutoencoder:
                encodings = self.model.encoder(torch.tensor(self.data).float().permute(0, 2, 1))
            else:
                encodings = self.model.encoder(torch.tensor(self.data).float())
        return encodings

    def euclidian_distance(self):
        """
        Calculates the euclidian distance between the latent representation of each stock
        :return: the euclidian distance between the latent representation of each stock
        """

        return get_distance_matrix(self.model, self.data, self.latent_size)

    def sample_trajectory_pairs(self, euclidian_distance):
        """
        Samples pairs of trajectories from the latent space
        :param euclidian_distance: Euclidean distance between the latent representation of each stock
        :return: close_pairs, all_pairs
        """

        # Compute delta
        delta = np.percentile(euclidian_distance, self.p)

        # Create a mask for pairs with distance in the latent space less than delta
        close_pairs_mask = np.triu(euclidian_distance < delta, k=1)

        # Get the indices of the close pairs and all pairs
        close_pairs_indices = np.column_stack(np.where(close_pairs_mask))
        all_pairs_indices = np.column_stack(np.where(np.triu(np.ones_like(euclidian_distance), k=1)))

        # Randomly sample S pairs of trajectories from the close pairs and all pairs
        close_pairs = close_pairs_indices[
            np.random.choice(close_pairs_indices.shape[0], self.batch_size, replace=False)]
        all_pairs = all_pairs_indices[np.random.choice(all_pairs_indices.shape[0], self.batch_size, replace=False)]

        return close_pairs, all_pairs

    @staticmethod
    @torch.jit.script
    def warping_distance_torch(t_A, t_B, alpha, gamma, epsilon):
        """
        Calculates the warping distance between two time series using pytorch
        :param t_A: Time series A
        :param t_B: Time series B
        :param alpha: Parameter alpha
        :param gamma: Parameter gamma
        :param epsilon: Parameter epsilon
        :return: Warping distance between the two time series
        """

        n, m = len(t_A), len(t_B)
        cost_matrix = torch.zeros((n + 1, m + 1))

        pairwise_euc = torch.cdist(t_A.unsqueeze(0), t_B.unsqueeze(0), p=2.0).squeeze()

        cost_matrix[1:, 0] = float('inf')
        cost_matrix[0, 1:] = float('inf')

        c1_factor = epsilon / (1 - epsilon)
        c2_factor = alpha / (1 - alpha)

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                c1 = c1_factor * torch.tanh(pairwise_euc[i - 1, j - 1] / c1_factor)
                c2 = c2_factor * (c1_factor * torch.tanh(pairwise_euc[i - 1, j - 1] / c1_factor)) + gamma
                updated_cost = c1 + cost_matrix[i - 1, j - 1]

                if i > 1:
                    updated_cost = torch.min(updated_cost, c2 + cost_matrix[i - 1, j])
                if j > 1:
                    updated_cost = torch.min(updated_cost, c2 + cost_matrix[i, j - 1])

                cost_matrix[i, j] = updated_cost.item()

        return cost_matrix[-1, -1]

    @staticmethod
    def warping_distance_torch_wrapper(args):
        """
        Wrapper for the warping distance function to be used with multiprocessing
        :param args: t_i, t_j, encodings, alpha, gamma, epsilon
        :return: The warping distance between the two time series
        """
        t_i, t_j, encodings, alpha, gamma, epsilon = args
        return AutoWarp.warping_distance_torch(encodings[t_i], encodings[t_j], alpha, gamma, epsilon)

    @staticmethod
    @jit(nopython=True)
    def warping_distance_numpy(t_A, t_B, alpha, gamma, epsilon):
        """
        Calculates the warping distance between two time series using numba (uses JIT compilation)
        :param t_A: Time series A
        :param t_B: Time series B
        :param alpha: Parameter alpha
        :param gamma: Parameter gamma
        :param epsilon: Parameter epsilon
        :return: The warping distance between the two time series
        """
        n, m = len(t_A), len(t_B)
        cost_matrix = np.zeros((n + 1, m + 1))

        for i in range(1, n + 1):
            cost_matrix[i, 0] = np.inf
        for j in range(1, m + 1):
            cost_matrix[0, j] = np.inf

        euclidean_distances = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                euclidean_distances[i, j] = np.linalg.norm(t_A[i] - t_B[j])
        sigma = lambda x, y: y * np.tanh(x / y)

        # Calculate the cost matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                c1 = sigma(euclidean_distances[i - 1, j - 1], epsilon / (1 - epsilon))
                c2 = alpha / (1 - alpha) * sigma(euclidean_distances[i - 1, j - 1], epsilon / (1 - epsilon)) + gamma
                updated_cost = c1 + cost_matrix[i - 1, j - 1]

                if i > 1:
                    updated_cost = min([updated_cost, c2 + cost_matrix[i - 1, j]])
                if j > 1:
                    updated_cost = min([updated_cost, c2 + cost_matrix[i, j - 1]])

                cost_matrix[i, j] = updated_cost

        return cost_matrix[-1, -1]

    @staticmethod
    def warping_distance_numpy_wrapper(args):
        """
        Wrapper for the numpy warping distance function
        :param args: i, j, encodings, alpha_numpy, gamma_numpy, epsilon_numpy
        :return: i, j, distance
        """
        i, j, encodings, alpha_numpy, gamma_numpy, epsilon_numpy = args
        distance = AutoWarp.warping_distance_numpy(encodings[i], encodings[j], alpha_numpy, gamma_numpy, epsilon_numpy)
        return i, j, distance

    def compute_gradients_and_beta_hat_torch(self, encodings, P_c, P_all, num_workers=None):
        """
        Computes gradients and updates alpha, gamma, epsilon and beta_hat
        :param encodings: Latent represenation of the data
        :param P_c: Pairs of data points with euclidian distance in the latent space less than delta
        :param P_all: Pairs of data points with any euclidian distance in the latent space
        :param num_workers: Number of workers to use for parallel computation
        :return: alpha, gamma, epsilon, beta_hat
        """

        tasks_c = [(t_i, t_j, encodings, self.alpha, self.gamma, self.epsilon) for t_i, t_j in P_c]
        tasks_all = [(t_i, t_j, encodings, self.alpha, self.gamma, self.epsilon) for t_i, t_j in P_all]

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            results_c = list(executor.map(AutoWarp.warping_distance_torch_wrapper, tasks_c))
            results_all = list(executor.map(AutoWarp.warping_distance_torch_wrapper, tasks_all))

        beta_hat_numerator = sum(results_c)
        beta_hat_denominator = sum(results_all)

        beta_hat = beta_hat_numerator.detach() / beta_hat_denominator.detach()
        beta_hat.requires_grad = True
        self.optimizer.zero_grad()
        beta_hat.backward()
        self.optimizer.step()

        return self.alpha, self.gamma, self.epsilon, beta_hat

    def learn_metric(self):
        """
        Learn the metric parameters alpha, gamma, epsilon and beta_hat
        :return: Learned alpha, gamma, epsilon and beta_hat
        """

        iteration = 0
        convergence = False

        # Setup calculations
        encodings = self.encodings()
        euclidian_distance = self.euclidian_distance()

        while not convergence and iteration < self.max_iterations:

            # Sample S pairs of trajectories from the euclidian distance matrix
            close_pairs, all_pairs = self.sample_trajectory_pairs(euclidian_distance)

            # Compute gradients and beta_hat
            if iteration > 0:
                beta_hat_old = beta_hat
            beta_hat, self.alpha, self.gamma, self.epsilon = self.compute_gradients_and_beta_hat_torch(
                encodings,
                close_pairs,
                all_pairs)

            iteration += 1

            # Check for convergence in beta_hat
            if iteration > 1:
                if abs(beta_hat - beta_hat_old) < 0.0001:
                    convergence = True

    def create_distance_matrix(self, num_workers=None):
        """
        Creates a distance matrix using the learned metric
        :param num_workers: Number of workers for parallel processing
        :return: Distance matrix
        """
        encodings = self.encodings().detach().numpy()
        num_trajectories = encodings.shape[0]
        distance_matrix = np.zeros((num_trajectories, num_trajectories))

        # Convert to numpy
        alpha_numpy = self.alpha.item()
        gamma_numpy = self.gamma.item()
        epsilon_numpy = self.epsilon.item()

        tasks = [(i, j, encodings, alpha_numpy, gamma_numpy, epsilon_numpy)
                 for i in range(num_trajectories)
                 for j in range(i + 1, num_trajectories)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(AutoWarp.warping_distance_numpy_wrapper, tasks))

        for i, j, distance in results:
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

        return distance_matrix
