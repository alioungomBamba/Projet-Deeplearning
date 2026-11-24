from pathlib import Path
from dataset import get_dataloader
import matplotlib.pyplot as plt
from model import DenoisingAutoencoder
import torch
import torch.nn as nn
import torch.optim as optim


def train_one_epoch(model, dataloader, criterion, optimizer, device):
  model.train()
  running_loss = 0.0

  for noisy_images, clean_images in dataloader:
    noisy_images = noisy_images.to(device)
    clean_images = clean_images.to(device)

    optimizer.zero_grad()
    outputs = model(noisy_images)
    loss = criterion(outputs, clean_images)
    loss.backward()
    optimizer.step()

    running_loss += loss.item() * noisy_images.size(0)

  return running_loss / len(dataloader.dataset)


def save_comparison(
    model, dataloader, device, noise_factor=0.3, out_path="resultats.png"
):
  model.eval()
  noisy_batch, clean_batch = next(iter(dataloader))
  noisy_batch = noisy_batch.to(device)
  clean_batch = clean_batch.to(device)

  with torch.no_grad():
    denoised_batch = model(noisy_batch)

  # Affiche jusqu'à 3 exemples (ou moins si le batch est plus petit)
  n_samples = min(3, noisy_batch.size(0))
  fig, axes = plt.subplots(n_samples, 3, figsize=(9, 3 * n_samples))
  if n_samples == 1:
    axes = axes.reshape(1, 3)

  for i in range(n_samples):
    noisy_img = noisy_batch[i].permute(1, 2, 0).cpu().numpy()
    denoised_img = denoised_batch[i].permute(1, 2, 0).cpu().numpy()
    clean_img = clean_batch[i].permute(1, 2, 0).cpu().numpy()

    axes[i, 0].imshow(noisy_img)
    axes[i, 0].set_title(f"Bruitée (σ={noise_factor})")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(denoised_img)
    axes[i, 1].set_title("Débruitée (IA)")
    axes[i, 1].axis("off")

    axes[i, 2].imshow(clean_img)
    axes[i, 2].set_title("Originale")
    axes[i, 2].axis("off")

  plt.tight_layout()
  plt.savefig(out_path, dpi=150)
  plt.close(fig)
  print(f"Comparaison enregistrée dans : {out_path}")


def main():
  # Paramètres (faciles à adapter pour les tests vs l'entraînement complet)
  image_dir = "./test"
  batch_size = 4
  image_size = 128
  noise_factor = 0.3
  epochs = 10
  learning_rate = 1e-3

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("Appareil utilisé :", device)

  dataloader = get_dataloader(
      image_dir=image_dir,
      batch_size=batch_size,
      image_size=image_size,
      noise_factor=noise_factor,
      shuffle=True,
      num_workers=0,  # Évite les soucis de multiprocessus sous WSL/Colab
  )

  model = DenoisingAutoencoder().to(device)
  criterion = nn.MSELoss()
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)

  losses = []
  for epoch in range(1, epochs + 1):
    epoch_loss = train_one_epoch(
        model, dataloader, criterion, optimizer, device
    )
    losses.append(epoch_loss)
    print(f"Époque {epoch:02d}/{epochs:02d} — Loss MSE : {epoch_loss:.6f}")

  # Sauvegarde des poids du modèle
  Path("checkpoints").mkdir(exist_ok=True)
  weights_path = "checkpoints/denoising_autoencoder.pth"
  torch.save(model.state_dict(), weights_path)
  print(f"Modèle sauvegardé dans : {weights_path}")

  # Courbe de loss
  plt.figure(figsize=(6, 4))
  plt.plot(range(1, epochs + 1), losses, marker="o")
  plt.title("Évolution de la perte (MSE) par époque")
  plt.xlabel("Époque")
  plt.ylabel("Loss MSE")
  plt.grid(True)
  plt.tight_layout()
  plt.savefig("loss_curve.png", dpi=150)
  plt.close()
  print("Courbe de loss enregistrée dans : loss_curve.png")

  # Image comparative avant / après
  save_comparison(
      model, dataloader, device, noise_factor=noise_factor, out_path="resultats.png"
  )


if __name__ == "__main__":
  main()