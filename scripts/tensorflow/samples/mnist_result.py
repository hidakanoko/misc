pi@raspberrypi:~/progs/tensorflow/samples $ date; python3 mnist_0817.py; date 
2018年  8月 24日 金曜日 10:02:26 JST
/usr/lib/python3.5/importlib/_bootstrap.py:222: RuntimeWarning: compiletime version 3.4 of module 'tensorflow.python.framework.fast_tensor_util' does not match runtime version 3.5
  return f(*args, **kwds)
/usr/lib/python3.5/importlib/_bootstrap.py:222: RuntimeWarning: builtins.type size changed, may indicate binary incompatibility. Expected 432, got 412
  return f(*args, **kwds)
WARNING:tensorflow:From mnist_0817.py:11: read_data_sets (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Instructions for updating:
Please use alternatives such as official/mnist/dataset.py from tensorflow/models.
WARNING:tensorflow:From /home/pi/.local/lib/python3.5/site-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:260: maybe_download (from tensorflow.contrib.learn.python.learn.datasets.base) is deprecated and will be removed in a future version.
Instructions for updating:
Please write your own downloading logic.
WARNING:tensorflow:From /home/pi/.local/lib/python3.5/site-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:262: extract_images (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Instructions for updating:
Please use tf.data to implement this functionality.
Extracting MNIST_data/train-images-idx3-ubyte.gz
WARNING:tensorflow:From /home/pi/.local/lib/python3.5/site-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:267: extract_labels (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Instructions for updating:
Please use tf.data to implement this functionality.
Extracting MNIST_data/train-labels-idx1-ubyte.gz
WARNING:tensorflow:From /home/pi/.local/lib/python3.5/site-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:110: dense_to_one_hot (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Instructions for updating:
Please use tf.one_hot on tensors.
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
WARNING:tensorflow:From /home/pi/.local/lib/python3.5/site-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:290: DataSet.__init__ (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Instructions for updating:
Please use alternatives such as official/mnist/dataset.py from tensorflow/models.
step 0, accuracy 415 (0.08)
step 500, accuracy 4670 (0.93)
step 1000, accuracy 4767 (0.95)
step 1500, accuracy 4814 (0.96)
step 2000, accuracy 4833 (0.97)
step 2500, accuracy 4863 (0.97)
step 3000, accuracy 4864 (0.97)
step 3500, accuracy 4874 (0.97)
step 4000, accuracy 4884 (0.98)
step 4500, accuracy 4899 (0.98)
step 5000, accuracy 4887 (0.98)
2018-08-24 10:10:00.954325: W tensorflow/core/framework/allocator.cc:108] Allocation of 67600000 exceeds 10% of system memory.
2018-08-24 10:10:02.154028: W tensorflow/core/framework/allocator.cc:108] Allocation of 67600000 exceeds 10% of system memory.
2018-08-24 10:10:02.154028: W tensorflow/core/framework/allocator.cc:108] Allocation of 67600000 exceeds 10% of system memory.
test accuracy 9790 (0.98)
2018年  8月 24日 金曜日 10:10:08 JST
