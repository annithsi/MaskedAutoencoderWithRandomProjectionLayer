import random
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm.auto import tqdm
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AutoEnc(nn.Module):
  def __init__(self):
      super(AutoEnc, self).__init__()
      #  6 original shape + 3 projected + 1 mask index = 10
      self.fc1 = nn.Linear(10, 5)
      self.fc2 = nn.Linear(5, 5)
      self.fc3 = nn.Linear(5, 6)

  def forward(self, x):
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = self.fc3(x)
      return x


class DimReduce(nn.Module):
  def __init__(self):
      super(DimReduce, self).__init__()
      #  6 original shape + 1 mask index = 7
      self.fc1 = nn.Linear(7, 5)
      self.fc2 = nn.Linear(5, 3)

  def forward(self, x):
      x = F.relu(self.fc1(x))
      x = self.fc2(x)
      return x


@dataclass(frozen=True)
class Config:
  mask_prob: float  # 0.0 - 1.0
  mask_time: float  # Mask time sec (Default: 0.4)
  input_feature_size: int  # Dimension of input.
  stride_time: float  # stride_time sec.
  code_book_size: int  # Dimension of code book (Default: 16)
  num_code_books: int  # Number of code books (Default: 8192)
  num_temporal_dimension_reduction_steps: int  # Number of temporal dimension reduction steps by the encoder
  encoder_hidden_size: int  # Number of encoder output dimensions


class RandomProjectionQuantizer(nn.Module):
  def __init__(self, config: Config):
    super().__init__()
    self.random_projection = nn.Linear(
      config.input_feature_size * config.num_temporal_dimension_reduction_steps, config.code_book_size, bias=False
    )
    nn.init.xavier_uniform_(self.random_projection.weight)

    self.code_book = nn.Parameter(torch.randn(config.num_code_books, config.code_book_size))

    self.random_projection.weight.requires_grad = False
    self.code_book.requires_grad = False

  @torch.no_grad()
  def forward(self, input_values: torch.Tensor, mask_time_indices: torch.Tensor) -> torch.Tensor:
    """
    Args:
        input_values (torch.Tensor): with shape `(B, L, D)`
        mask_time_indices (torch.Tensor): with shape `(B, L)`

    Returns:
        torch.Tensor with shape `(N)`

    """
    targets = self.random_projection(input_values[mask_time_indices == 1]).unsqueeze(1)

    # Compute l2 norm targets and code vectors
    vector_distances = torch.linalg.vector_norm(targets - self.code_book, dim=-1)

    labels = torch.argmin(vector_distances, dim=-1)

    return labels


class BestRqFramework(nn.Module):
  def __init__(self, config: Config, encoder: nn.Module):
    super().__init__()
    self.K = config.num_temporal_dimension_reduction_steps
    self.layer_norm = nn.LayerNorm(config.input_feature_size)
    self.random_projection_quantizer = RandomProjectionQuantizer(config)
    self.encoder = encoder
    self.config = config
    self.out_linear = nn.Linear(config.encoder_hidden_size, config.num_code_books)
    self.num_time_steps = int(config.mask_time // (config.stride_time * self.K))

  def forward(self, input_values: torch.Tensor, input_lengths: torch.Tensor):
    """
    Args:
        input_values (torch.Tensor): with shape `(B, T, D)`
        input_lengths (torch.Tensor): with shape `(B)`

    Returns:

    """
    batch_size, num_steps, hidden_size = input_values.size()

    input_values = self.layer_norm(input_values)

    if not num_steps % self.config.num_temporal_dimension_reduction_steps == 0:
      transformed_num_steps = (num_steps // self.K + 1) * self.K
      padding = torch.zeros(
        batch_size, transformed_num_steps - num_steps, hidden_size, device=input_values.device
      )
      input_values = torch.cat([input_values, padding], dim=1)
      num_steps = transformed_num_steps

    # Reshape to number of encoder out steps
    input_values = input_values.view(batch_size, -1, self.K * hidden_size)
    quantized_input_lengths = input_lengths // self.K - 1

    masked_input_values, time_mask_indices = self.masking(input_values.clone(), quantized_input_lengths)
    masked_input_values = masked_input_values.view(batch_size, num_steps, hidden_size)

    labels = self.random_projection_quantizer(input_values, time_mask_indices)

    encoder_out = self.encoder(masked_input_values, input_lengths)

    targets = encoder_out[time_mask_indices]
    targets_out = self.out_linear(targets)

    return targets_out, labels

  def masking(self, input_values: torch.Tensor, input_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        input_values (torch.Tensor): with shape `(B, L, D)`
        input_lengths (torch.Tensor): with shape `(B)'

    Returns:
        tuple(
        torch.Tensor with shape `(B, L, D)`
        torch.Tensor with shape `(B, L)`
        )
    """
    batch_size, num_steps, hidden_size = input_values.size()

    # non mask: 0, maks: 1
    time_mask_indices = torch.zeros(
      batch_size, num_steps + self.num_time_steps,
      device=input_values.device, dtype=torch.bool
    )

    for batch in range(batch_size):
      time_mask_idx_candidates = list(range(int(input_lengths[batch])))
      k = int(self.config.mask_prob * input_lengths[batch])
      start_time_mask_idx_array = torch.tensor(
        random.sample(time_mask_idx_candidates, k=k), device=input_values.device, dtype=torch.long
      )

      for i in range(self.num_time_steps):
        time_mask_indices[batch, start_time_mask_idx_array + i] = 1

    time_mask_indices = time_mask_indices[:, :-self.num_time_steps]
    num_masks = sum(time_mask_indices.flatten())

    # Replace to random value where mask
    random_values = torch.normal(mean=0, std=0.1, size=(num_masks, hidden_size), device=input_values.device)
    input_values[time_mask_indices == 1] = random_values

    return input_values, time_mask_indices


def train_model(model,criterion,optimizer,dataloaders,epochs,check_every=None,earlyStopping=False):

  print("training model")
  optimizer.zero_grad()

  if not check_every:
      check_every = int(epochs / 10) if epochs > 10 else 1

  phases = dataloaders.keys()
  valExists = True if "val" in phases else False
  avg_loss = {phase:0 for phase in phases}
  avg_losses = {phase:[] for phase in phases}

  for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times

    batchLoss = {phase:[] for phase in phases}

     # Each epoch has a training and validation phase
    for phase in phases:
      if phase == "train":  model.train()  # Set model to training mode
      else: model.eval()   # Set model to evaluate mode

      for i, (inputBatch,labelBatch) in enumerate(tqdm(dataloaders[phase], desc=phase, leave=False)):

          inputBatch = inputBatch.to(device).float()
          labelBatch = labelBatch.to(device).float()

          # forward
          with torch.set_grad_enabled(not phase=="val"):
            outputBatch = model(inputBatch)
          loss = criterion(outputBatch, labelBatch)
          batchLoss[phase].append(loss.item())

          # backward + optimize only if in training phase
          if phase == "train":
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


    for phase in phases : avg_loss[phase] = np.mean(batchLoss[phase])

    phase = "val" if valExists else "train"
    if epoch > 0:
      if avg_loss[phase] < min(avg_losses[phase]):
        best_params = deepcopy(model.state_dict())
        best_epoch, best_loss = epoch, avg_loss[phase]
    else:
      best_params = deepcopy(model.state_dict())
      best_epoch, best_loss = epoch, avg_loss[phase]
      movAvg_old = avg_loss[phase]

    for phase in phases : avg_losses[phase].append(avg_loss[phase])

    # print statistics
    if epoch % check_every == check_every - 1:
      print("epoch: %d" % (epoch + 1), end="  | ")
      for phase in phases:
        print("%s loss: %.3f" % (phase, avg_loss[phase]), end=", ")
      if check_every > 1:
        print(" | ", end='')
        for phase in phases:
          print("avg %s loss: %.3f" % (phase, np.mean(avg_losses[phase][epoch+1-check_every:epoch+1])), end=", ")
      if valExists:
        movAvg_new = np.mean(avg_losses["val"][epoch+1-check_every:epoch+1])

      if (valExists) and earlyStopping:
        if movAvg_old < movAvg_new:   break
        else:   movAvg_old = movAvg_new

  end_params = deepcopy(model.state_dict())
  print('Finished Training')
  for phase in phases:  plt.plot(avg_losses[phase], label=phase+" loss")
  # plt.plot([best_loss]*epoch, linestyle='dashed')
  plt.plot(best_epoch, best_loss, 'o')
  plt.xlabel("epoch")
  plt.ylabel("loss")
  plt.legend()
  plt.show()

  return best_params, end_params


def evaluate(net, inputs):
  net.eval()
  output_pred = []

  with torch.no_grad():
    for input_ in tqdm(inputs):
      output = net(torch.from_numpy(input_).unsqueeze(0).float().to(device)).cpu().numpy()
      output_pred.append(output[0])

  return np.array(output_pred)
