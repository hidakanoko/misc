import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import urllib.request

# 参考資料
# http://docs.fabo.io/tensorflow/model_logstic/tensorflow_three_classification_last.html
# https://qiita.com/TomokIshii/items/92a266b805d7eee02b1d
# https://www.easy-tensorflow.com/tf-tutorials/autoencoders/noise-removal

# データファイルのダウンロード
train_datafile = './iris_training.csv'
if not os.path.exists(train_datafile):
    urllib.request.urlretrieve(
        'http://download.tensorflow.org/data/iris_training.csv',
        train_datafile)

test_datafile = './iris_test.csv'
if not os.path.exists(test_datafile):
    urllib.request.urlretrieve(
        'http://download.tensorflow.org/data/iris_test.csv',
        test_datafile)

# CSVファイルの読み込み
train_dataset = np.genfromtxt(
    train_datafile,
    delimiter=',',
    skip_header=1,
    dtype=[float, float, float, float, int])
test_dataset = np.genfromtxt(
    test_datafile,
    delimiter=',',
    skip_header=1,
    dtype=[float, float, float, float, int])

# shuffle
np.random.shuffle(train_dataset)
np.random.shuffle(test_dataset)

# 読み込んだものをラベルとデータに分割
input_units = 4
output_units = 3


def get_labels(dataset):
    raw_labels = [item[input_units] for item in dataset]
    # one hotにして返す
    return np.eye(output_units)[raw_labels]
    # 以下でも同じ
    # return np.array(tf.Session().run(tf.one_hot(raw_labels, output_units)))


def get_data(dataset):
    raw_data = [list(item)[:input_units] for item in dataset]
    return np.array(raw_data)

train_labels = get_labels(train_dataset)
test_labels = get_labels(test_dataset)
train_data = get_data(train_dataset)
test_data = get_data(test_dataset)
print(train_labels.shape)
print(test_labels.shape)
print(train_data.shape)
print(test_data.shape)

# ミニバッチに分割
batch_size = 32
train_labels_batch = []
train_data_batch = []
for i in range(0, train_labels.shape[0], batch_size):
    train_labels_batch.append(train_labels[i:i + batch_size])
    train_data_batch.append(train_data[i:i + batch_size])
batch_split = len(train_labels_batch)

# predict用データ
predict_data = np.array([
    [5.1, 3.3, 1.7, 0.5],
    [5.9, 3.0, 4.2, 1.5],
    [6.9, 3.1, 5.4, 2.1]
])
print(predict_data.shape)

# データを格納するplaceholder
X = tf.placeholder(tf.float32, shape=(None, input_units))
# ラベルを格納するplaceholder
y_ = tf.placeholder(tf.float32, shape=(None, output_units))

# 隠れ層と出力層の定義
n = 0


def layer(layer, units, hidden):
    # 各層の重み付けとバイアスを保持する変数
    global n
    with tf.variable_scope('layer' + str(n), reuse=False):
        w = tf.get_variable(
            'w',
            shape=(layer.get_shape()[1], units),
            initializer=tf.glorot_normal_initializer())
        b = tf.get_variable(
            'b',
            shape=(units, ),
            initializer=tf.zeros_initializer())
        # 次は上と大体同じだが、セッション内で現在値を参照できない
        # b = tf.Variable(tf.zeros([units]))
        n += 1
    # 重み付けを掛けてバイアスを足す
    f = tf.matmul(layer, w) + b
    # 活性化関数
    if hidden:
        p = tf.nn.relu(f)
        # 次はtf.nn.reluと同じ
        # p = tf.maximum(f, 0)
    else:
        # tf.losses.softmax_cross_entropyを使う場合、出力層のsoftmax不要
        # p = tf.nn.softmax(f)
        p = f
    return p

h1 = layer(X, 30, True)
h2 = layer(h1, 10, True)
p = layer(h2, output_units, False)

# trainに使用する誤差関数とトレーニングアルゴリズム(optimizer)
loss = tf.losses.softmax_cross_entropy(onehot_labels=y_, logits=p)
# cross entropyは以下で書くこともできる
# ただしtf.losses.softmax_cross_entropyで行われる、出力層の活性化関数softmaxは追加する必要あり
# loss = -tf.reduce_mean(y_ * tf.log(p))
optimizer = tf.train.AdamOptimizer()
# optimizerの自作は非常に面倒だと思われるのでここでは行わない。調べたところでは以下の通り
# https://qiita.com/yuishihara/items/73e8f8c4a30b8148d9fc
global_step = tf.train.get_or_create_global_step()
train = optimizer.minimize(loss, global_step=global_step)
# optimizerを利用する際minimizeを呼ぶ方法以外に以下のようにapply_gradientsを呼ぶ方法がある
# https://www.tensorflow.org/api_docs/python/tf/train/Optimizer
# params = tf.trainable_variables()
# gradients = tf.gradients(loss, params)
# train = optimizer.apply_gradients(
#     zip(gradients, params), global_step=global_step)

# evaluateとpredict
pred = tf.argmax(p, 1)
accuracy = tf.reduce_mean(
    tf.cast(tf.equal(pred, tf.argmax(y_, 1)), tf.float32))

train_loss_results = []
train_accuracy_results = []

# 学習の実行
with tf.Session() as sess:
    print(type(sess))
    sess.run(tf.global_variables_initializer())
    for i in range(2001):
        # train
        data = train_data_batch[i % batch_split]
        labels = train_labels_batch[i % batch_split]
        sess.run(train, feed_dict={X: data, y_: labels})
        # trainデータでevaluate
        train_loss, train_acc = sess.run(
            [loss, accuracy],
            feed_dict={X: train_data, y_: train_labels})
        train_loss_results.append(train_loss)
        train_accuracy_results.append(train_acc)
        if i % 500 == 0:
            print("Step: %d" % i)
            print("loss: %f, acc: %f" % (train_loss, train_acc))
            # 出力層の重み付けとバイアスを出力する
            with tf.variable_scope('layer2', reuse=True):
                w, b = sess.run(
                    [tf.get_variable('w', shape=(10, output_units)),
                     tf.get_variable('b', shape=(output_units, ))])
                print("weight: %s\nbias: %s" % (w, b))

    # matplotlib
    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle('Training Metrics')

    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].plot(train_loss_results)

    axes[1].set_ylabel("Accuracy", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].plot(train_accuracy_results)

    plt.show()

    # testデータでevaluate
    test_loss, test_acc = sess.run(
        [loss, accuracy],
        feed_dict={X: test_data, y_: test_labels})
    print("[Test] loss: %f, acc: %f" % (test_loss, test_acc))

    # predict
    print("[Predict]: %s" % sess.run(pred, feed_dict={X: predict_data}))
    # すべてのsess.runはsess = tf.Session()ではなくwith tf.Session() as sess:にしているか、
    # tf.Session()の代わりにtf.InteractiveSession()を使っている場合、
    # Tensorクラスのevalを使って書くこともできる
    # print("[Predict]: %s" % pred.eval(feed_dict={X: predict_data}))
