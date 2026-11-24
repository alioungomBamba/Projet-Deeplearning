from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def add_gaussian_noise(images: torch.Tensor, noise_factor: float = 0.3):
  """Ajoute un bruit gaussien aléatoire à un tenseur d'images normalisé entre 0 et 1."""
  noise = torch.randn_like(images) * noise_factor
  noisy_images = torch.clamp(images + noise, 0.0, 1.0)
  return noisy_images, images


class FFHQUnsupervisedDenoisingDataset(Dataset):
  """Charge un dossier d'images (ex. FFHQ 128x128) et génère des paires (bruitée, propre)."""

  def __init__(
      self,
      image_dir: str,
      image_size: int = 128,
      noise_factor: float = 0.3,
      max_images: int | None = None,
  ):
    self.image_dir = Path(image_dir)
    self.image_size = image_size
    self.noise_factor = noise_factor

    # Récupère les images courantes (PNG, JPG, JPEG)
    extensions = (
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.jfif",
    "*.webp",
    "*.PNG",
    "*.JPG",
    "*.JPEG",
    "*.JFIF",
    "*.WEBP",
)
    files = []
    for ext in extensions:
      files.extend(self.image_dir.glob(ext))
    self.image_paths = sorted(files)

    # Pratique pour tester rapidement sur un petit échantillon avant l'entraînement complet
    if max_images is not None:
      self.image_paths = self.image_paths[:max_images]

    if len(self.image_paths) == 0:
      raise FileNotFoundError(
          f"Aucune image trouvée dans le dossier : {self.image_dir}"
      )

    self.transform = transforms.Compose([
        transforms.Resize((self.image_size, self.image_size)),
        transforms.ToTensor(),  # Convertit en tenseur (C, H, W) dans [0.0, 1.0]
    ])

  def __len__(self) -> int:
    return len(self.image_paths)

  def __getitem__(self, idx: int):
    img_path = self.image_paths[idx]
    with Image.open(img_path) as img:
      clean_image = self.transform(img.convert("RGB"))

    # Ajoute le bruit sur une seule image (C, H, W)
    noisy_image, clean_image = add_gaussian_noise(
        clean_image, noise_factor=self.noise_factor
    )
    return noisy_image, clean_image


def get_dataloader(
    image_dir: str,
    batch_size: int = 32,
    image_size: int = 128,
    noise_factor: float = 0.3,
    max_images: int | None = None,
    shuffle: bool = True,
    num_workers: int = 2,
) -> DataLoader:
  dataset = FFHQUnsupervisedDenoisingDataset(
      image_dir=image_dir,
      image_size=image_size,
      noise_factor=noise_factor,
      max_images=max_images,
  )
  return DataLoader(
      dataset,
      batch_size=batch_size,
      shuffle=shuffle,
      num_workers=num_workers,
      pin_memory=torch.cuda.is_available(),
  )


# Petit test local rapide si tu exécutes `python dataset.py` directement
if __name__ == "__main__":
  sample_dir = "./test"
  print(f"Test de chargement depuis : {sample_dir}")
  try:
    loader = get_dataloader(sample_dir, batch_size=8, max_images=16)
    noisy_batch, clean_batch = next(iter(loader))
    print("Batch bruité :", noisy_batch.shape, noisy_batch.dtype)
    print("Batch propre :", clean_batch.shape, clean_batch.dtype)
  except FileNotFoundError as e:
    print(e)
    print(
        "Crée ce dossier et mets-y quelques images de test pour vérifier"
        " localement."
    )