import torch

from torch.utils.data import Dataset


class DiacritizationDataset(Dataset):
  def __init__(self, file_path: str, characters: list[str], diacritics: list[str]):
    self.data = []
    self.characters = characters
    self.diacritics = diacritics

    with open(file_path) as fp:
      for line in fp:
        line = line.strip()

        if line:
          self.data.append(self._encode_line(line))

  def collate_fn(
    self,
    batch: tuple[torch.Tensor, torch.Tensor],
  ) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int]]:
    characters_batch, diacritics_batch = zip(*batch)

    return (
      pad_sequence(characters_batch, batch_first=True, padding_value=self.characters.index('<pad>')),
      pad_sequence(diacritics_batch, batch_first=True, padding_value=self.diacritics.index('<pad>')),
      [len(element) for element in characters_batch],
      [len(element) for element in diacritics_batch],
    )

  def __len__(self) -> int:
    return len(self.data)

  def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
    return self.data[index]

  def _encode_line(self, line: str) -> tuple[torch.Tensor, torch.Tensor]:
    current_diacritics = ''
    encoded_characters = []
    encoded_diacritics = []

    for character in line:
      if character in self.characters:
        encoded_characters.append(self.characters.index(character))

        encoded_diacritics.append(self.diacritics.index(current_diacritics))
        current_diacritics = ''
      else:
        current_diacritics += character

    encoded_diacritics.append(self.diacritics.index(current_diacritics))
    current_diacritics = ''

    return (
      torch.tensor(encoded_characters + [self.characters.index('<end>')], dtype=torch.long),
      torch.tensor(encoded_diacritics[1:] + [self.diacritics.index('<end>')], dtype=torch.long),
    )
