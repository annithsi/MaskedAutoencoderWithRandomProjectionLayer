import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn import random_projection


def mask_input(x: np.ndarray):
  """Randomly sets one of the channel values to -1."""
  num_samples, num_channels = x.shape
  mask = np.random.randint(0, num_channels, num_samples)
  x_masked = x.copy()
  x_masked[np.arange(num_samples), mask] = -1
  return x_masked, mask[:, None]


def insert_projections_and_mask(x: np.ndarray, n_components=2):
  x_masked, mask = mask_input(x)
  projector = random_projection.SparseRandomProjection(n_components=n_components)
  x_projected = projector.fit_transform(x)
  return np.hstack((x, mask, x_projected))


# custom dataset class
class mydataset(Dataset):
  def __init__(self, inputs, outputs):
    self.inputs = inputs
    self.outputs = outputs

  def __len__(self):
    return len(self.inputs)

  def __getitem__(self, index):
    input = self.inputs[index]
    output = self.outputs[index]
    return input, output


def get_loaders(inputs, outputs):
  train_inputs, val_inputs, train_outputs, val_outputs = train_test_split(inputs, outputs, test_size=0.25, random_state=0)
  train_dataset = mydataset(train_inputs, train_outputs)
  train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
  val_dataset = mydataset(val_inputs, val_outputs)
  val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
  dataloaders = {'train': train_loader, 'val': val_loader}
  return dataloaders
