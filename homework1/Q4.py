from Util import train_model
from Models import MANN2

itr_cnt = 40000
# Config = (K,N)
# K: num of samples per class
# N: num of classes
configs = [(1,5), (4,5)]

for config in configs:
  K, N = config
  config_name = f'K={K}&N={N}'
  print(f'Testing {config_name}')
  model = MANN2(N, K+1, cell_count=128)
  train_model(num_classes=N,
              num_samples=K,
              model=model,
              meta_batch_size=64,
              random_seed=1234,
              itr_cnt=itr_cnt,
              logDir='./models/question4',
              modelDir='./models/question4')