import numpy as np

from utils.data_loader import get_data_loader

from model.wgan import WGAN_CP

train_loader = get_data_loader('data/1.csv', 'Horizontal_vibration_signals', 35, 1)

