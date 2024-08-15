import os
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim

import sys
sys.path.append("../")

from modules import models, data_f


def main():
  # Read the data.
  script_dir = os.path.dirname(os.path.abspath(__file__))
  data_dir = os.path.join(script_dir, '..', 'data')
  data_file_path = os.path.join(data_dir, 'ADBench/datasets/Classical/2_annthyroid.npz')
  data = np.load(data_file_path, allow_pickle=True)
  x, y = data['X'], data['y']

  # Split x into x_normal and x_anomaly based on indices.
  normal_indices = np.where(y == 0)[0]
  anomaly_indices = np.where(y == 1)[0]
  x_normal = x[normal_indices]
  x_anomaly = x[anomaly_indices]

  # Add the mask indices and projections as an extra channel to x.
  x_normal_masked = data_f.insert_projections_and_mask(x_normal, n_components=3)
  x_anomaly_masked = data_f.insert_projections_and_mask(x_anomaly, n_components=3)

  # Create dataloaders and model followed by training.
  dataloaders = data_f.get_loaders(x_normal_masked, x_normal)
  model = models.AutoEnc()
  optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
  criterion = nn.MSELoss()
  best_params, end_params = models.train_model(model, criterion, optimizer, dataloaders, epochs=10)
  model.load_state_dict(best_params)

  predictions_normal = models.evaluate(model, x_normal_masked)
  err_normal = abs(predictions_normal - x_normal)
  predictions_anomaly = models.evaluate(model, x_anomaly_masked)
  err_anomaly = abs(predictions_anomaly - x_anomaly)

  # Visualize reconstruction loss for each channel using box plot.
  fig, axs = plt.subplots(1, 2)
  plt.sca(axs[0])
  plt.boxplot(err_normal)
  plt.title('Absolute reconstruction error for the normal class.')
  plt.sca(axs[1])
  plt.boxplot(err_anomaly)
  plt.title('Absolute reconstruction error for the anomaly class.')
  plt.show()


if __name__ == "__main__":
  main()
