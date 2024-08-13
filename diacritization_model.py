import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class DiacritizationModel(nn.Module):
  def __init__(
    self,
    input_size: int,
    embedding_size: int,
    hidden_size: int,
    output_size: int,
    num_layers: int = 2,
    dropout: float = 0.5,
    bidirectional: bool = True,
  ):
    super(DiacritizationModel, self).__init__()
    self.hidden_size = hidden_size
    self.bidirectional = bidirectional

    self.embedding = nn.Embedding(input_size, embedding_size)
    self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional)
    self.dropout = nn.Dropout(p=dropout)
    self.fc1 = nn.Linear(hidden_size * (2 if bidirectional else 1), hidden_size * (2 if bidirectional else 1))
    self.fc2 = nn.Linear(hidden_size * (2 if bidirectional else 1), hidden_size * (2 if bidirectional else 1))
    self.fc3 = nn.Linear(hidden_size * (2 if bidirectional else 1), output_size)

    self.relu = nn.ReLU()

  def forward(
    self,
    input: torch.Tensor,
    input_lengths: torch.Tensor,
    padding_value: int,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    embeds = self.embedding(input)
    packed_embedded = pack_padded_sequence(embeds, input_lengths, batch_first=True, enforce_sorted=False)

    output, (_hidden, _cell) = self.lstm(packed_embedded)
    output, output_lengths = pad_packed_sequence(output, batch_first=True, padding_value=padding_value)

    output = output.contiguous().view(-1, self.hidden_size * (2 if self.bidirectional else 1))

    output = self.dropout(output)

    output = self.relu(self.fc1(output))
    output = self.relu(self.fc2(output))
    output = self.fc3(output)

    return output, output_lengths
