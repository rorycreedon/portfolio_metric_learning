import torch
import torch.nn as nn
import torch.optim as optim
from tslearn.metrics import cdist_soft_dtw, cdist_soft_dtw_normalized
from scipy.spatial.distance import cdist


class LinearAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(LinearAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )
    
    def forward(self, x):
        encoding = self.encoder(x.permute(0, 2, 1))
        decoded = self.decoder(encoding).permute(0, 2, 1) # permute dimensions
        return decoded


class ConvAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_size, latent_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_size, hidden_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_size, input_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        encoding = self.encoder(x)
        decoded = self.decoder(encoding) # permute dimensions
        return decoded


class ConvLinearAutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, latent_size):
        super(ConvLinearAutoEncoder, self).__init__()
        self.linear_encoder1 = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU()
        )
        self.conv_encoder = nn.Sequential(
            nn.Conv1d(hidden_size1, hidden_size2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_size2, latent_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.linear_encoder2 = nn.Linear(latent_size, latent_size)

        self.linear_decoder1 = nn.Sequential(
            nn.Linear(latent_size, latent_size),
            nn.ReLU()
        )

        self.conv_decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_size, hidden_size2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_size2, hidden_size1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.linear_decoder2 = nn.Linear(hidden_size1, input_size)
    
    def encoder(self, x):
        encoding = self.linear_encoder1(x.permute(0, 2, 1))
        encoding = self.conv_encoder(encoding.permute(0, 2, 1))
        encoding = self.linear_encoder2(encoding.permute(0, 2, 1))
        return encoding
    
    def decoder(self, x):
        decoded = self.linear_decoder1(x)
        decoded = self.conv_decoder(decoded.permute(0, 2, 1))
        decoded = self.linear_decoder2(decoded.permute(0, 2, 1)).permute(0, 2, 1)
        return decoded

    def forward(self, x):
        encoding = self.encoder(x)
        decoded = self.decoder(encoding)
        return decoded


def train_autoencoder(autoencoder_class, input_size, hidden_size, latent_size, num_epochs, batch_size, data, lr=0.001, hidden_size2=None, verbose=True):

    # Setup model, loss function, optimizer
    if hidden_size2 is None:
        model = autoencoder_class(input_size, hidden_size, latent_size)
    else:
        model = autoencoder_class(input_size, hidden_size, hidden_size2, latent_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if verbose:
        print(autoencoder_class.__name__)

    # Train data
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i in range(0, data.shape[0], batch_size):
            inputs = torch.tensor(data[i:i+batch_size]).float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = torch.mean(criterion(outputs, inputs))
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        if verbose:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, running_loss / data.shape[0]))

    # Return trained model
    return model


def get_distance_matrix(model, data, latent_size, distance_metric="euclidean", gamma=1.0):

    with torch.no_grad():

        model.eval()
        if type(model) == LinearAutoencoder:
            encodings = model.encoder(torch.tensor(data).float().permute(0, 2, 1))
        else:
            encodings = model.encoder(torch.tensor(data).float())

        if distance_metric == "euclidean":
            encodings = encodings.reshape(data.shape[0], data.shape[2] * latent_size)  # Flatten the last two dimensions
            distances = cdist(encodings, encodings, metric='euclidean')
            return distances
        elif distance_metric == "soft_dtw":
            distances = cdist_soft_dtw(encodings, gamma=gamma)
            return distances
        elif distance_metric == "soft_dtw_normalized":
            distances = cdist_soft_dtw_normalized(encodings, gamma=gamma)
            return distances
        else:
            raise ValueError("Distance not recognized/implemented.")
