import pickle

from pathlib import Path

import torch
import torch.nn as nn

from tap import Tap
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm


from constants import CHARACTERS_FILE_PATH, TRAIN_FILE_PATH, VAL_FILE_PATH, DIACRITICS, DIACRITICS_COMBINATIONS, PAD, END
from diacritization_dataset import DiacritizationDataset
from diacritization_model import DiacritizationModel


class ShakkelhaArgumentParser(Tap):
  epochs: int = 5
  batch_size: int = 32
  embedding_size: int = 25
  hidden_size: int = 256


def main():
  args = ShakkelhaArgumentParser().parse_args()

  characters = load_characters()

  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  train_dataset = DiacritizationDataset(TRAIN_FILE_PATH, characters, DIACRITICS_COMBINATIONS)
  train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)

  val_dataset = DiacritizationDataset(VAL_FILE_PATH, characters, DIACRITICS_COMBINATIONS)
  val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=val_dataset.collate_fn)

  model = DiacritizationModel(len(characters), args.embedding_size, args.hidden_size, len(DIACRITICS_COMBINATIONS))
  model.to(device)

  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters())

  train(model, train_data_loader, val_data_loader, optimizer, criterion, device, args.epochs, characters)


def load_characters() -> list[str]:
  if not Path(CHARACTERS_FILE_PATH).exists():
    characters = sorted(list(set(open(TRAIN_FILE_PATH).read()) - set(DIACRITICS + ['\n'])) + [PAD, END])

    with open(CHARACTERS_FILE_PATH, 'wb') as f:
      pickle.dump(characters, f)
  else:
    with open(CHARACTERS_FILE_PATH, 'rb') as f:
      characters = pickle.load(f)

  return characters


def train(
  model: nn.Module,
  train_data_loader: DataLoader,
  val_data_loader: DataLoader,
  optimizer: torch.optim.Optimizer,
  criterion: nn.Module,
  device: str,
  epochs: int,
  characters: list[str],
) -> None:
  for epoch in tqdm(range(epochs), desc='Epochs'):
    for (
      batch_index,
      (
        characters_batch,
        diacritics_batch,
        characters_lengths,
        _diacritics_lengths,
      ),
    ) in enumerate(tqdm(train_data_loader, desc='Batches')):
      characters_batch = characters_batch.to(device)
      diacritics_batch = diacritics_batch.to(device)
      characters_lengths = torch.tensor(characters_lengths, dtype=torch.long)

      output, output_lengths = model(characters_batch, characters_lengths, characters.index('<pad>'))

      loss = criterion(output.view(-1, output.size(-1)), diacritics_batch.view(-1))

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if (batch_index + 1) % 1000 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_index + 1}/{len(train_data_loader)}], Loss: {loss.item():.4f}")

    model.eval()

    val_loss = 0

    with torch.no_grad():
      for (characters_batch, diacritics_batch, characters_lengths, _diacritics_lengths) in val_data_loader:
        characters_batch = characters_batch.to(device)
        diacritics_batch = diacritics_batch.to(device)
        characters_lengths = torch.tensor(characters_lengths, dtype=torch.long)

        output, output_lengths = model(characters_batch, characters_lengths, characters.index('<pad>'))
        loss = criterion(output.view(-1, output.size(-1)), diacritics_batch.view(-1))
        val_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{epochs}], Validation Loss: {val_loss / len(val_data_loader):.4f}")

    model.train()


if __name__ == '__main__':
    main()
