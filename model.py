import torch
import torch.nn as nn


class DenoisingAutoencoder(nn.Module):
  """Auto-encodeur convolutif pour le débruitage d'images RGB 128x128."""

  def __init__(self):
    super().__init__()

    # Encodeur : 128x128 -> 64x64 -> 32x32 -> 16x16
    self.encoder = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
    )

    # Décodeur : 16x16 -> 32x32 -> 64x64 -> 128x128
    self.decoder = nn.Sequential(
        nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
        ),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
        ),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(
            32, 3, kernel_size=3, stride=2, padding=1, output_padding=1
        ),
        nn.Sigmoid(),  # Ramène les pixels dans [0.0, 1.0]
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    latent = self.encoder(x)
    reconstructed = self.decoder(latent)
    return reconstructed


if __name__ == "__main__":
  model = DenoisingAutoencoder()
  dummy_input = torch.randn(7, 3, 128, 128)
  output = model(dummy_input)
  print("Entrée :", dummy_input.shape)
  print("Sortie :", output.shape)