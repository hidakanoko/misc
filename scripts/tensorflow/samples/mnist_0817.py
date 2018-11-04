import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 参考資料
# https://deepinsider.jp/tutor/introtensorflow/buildcnn

# 毎回同じ結果になるように固定値で乱数を初期化
tf.set_random_seed(1)

# MNISTデータセットの読み込み
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

n = 0


# 畳み込み層
def convolutional_layer(layer, size, strides, padding, prob):
    global n
    with tf.variable_scope('layer' + str(n), reuse=False):
        w = tf.get_variable(
            'w',
            shape=size,
            initializer=tf.glorot_normal_initializer())
        b = tf.get_variable(
            'b',
            shape=[size[3]],
            initializer=tf.zeros_initializer())
        n += 1
    return tf.nn.relu(
        tf.nn.conv2d(
            tf.nn.dropout(layer, prob), w,
            strides=strides, padding=padding) + b)


# プーリング層
def pooling_layer(layer, size, strides, padding, prob):
    return tf.nn.max_pool(
        tf.nn.dropout(layer, prob), ksize=size,
        strides=strides, padding=padding)


# 全結合層
def fully_connected_layer(layer, units, prob, hidden):
    global n
    with tf.variable_scope('layer' + str(n), reuse=False):
        w = tf.get_variable(
            'w',
            shape=[layer.get_shape()[1], units],
            initializer=tf.glorot_normal_initializer())
        b = tf.get_variable(
            'b',
            shape=[units],
            initializer=tf.zeros_initializer())
        n += 1
    f = tf.matmul(tf.nn.dropout(layer, prob), w) + b
    if hidden:
        return tf.nn.relu(f)
    return f


# ドロップアウトさせない率を格納するplaceholder
prob_input = tf.placeholder(tf.float32)
prob_common = tf.placeholder(tf.float32)
prob_output = tf.placeholder(tf.float32)

# データを格納するplaceholder
x = tf.placeholder(tf.float32)
# ラベルを格納するplaceholder
y_ = tf.placeholder(tf.float32)

# 畳み込み処理やプーリング処理に渡すには4次元にreshapeする
layer_in = tf.reshape(x, [-1, 28, 28, 1])

# 畳み込み層とプーリング層の定義
kernel_count1 = 10
kernel_count2 = kernel_count1 * 2

layer_c1 = convolutional_layer(
    layer_in,
    [4, 4, 1, kernel_count1],
    [1, 2, 2, 1],
    'VALID',
    prob_input)
layer_c2 = convolutional_layer(
    layer_c1,
    [3, 3, kernel_count1, kernel_count2],
    [1, 2, 2, 1],
    'VALID',
    prob_common)
layer_c3 = pooling_layer(
    layer_c2,
    [1, 2, 2, 1],
    [1, 2, 2, 1],
    'VALID',
    prob_common)

# 全結合処理に渡すには2次元にreshapeする
layer_f0 = tf.reshape(
    layer_c3, [-1, layer_c3.shape[1] * layer_c3.shape[2] * layer_c3.shape[3]])

# 全結合層の定義
layer_f1 = fully_connected_layer(layer_f0, 40, prob_common, True)
layer_out = fully_connected_layer(layer_f1, 10, prob_output, False)

# 誤差関数とトレーニングアルゴリズムとevaluate
loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=layer_out)
optimizer = tf.train.AdamOptimizer().minimize(loss)
pred = tf.argmax(layer_out, 1)
accuracy = tf.reduce_sum(
    tf.cast(tf.equal(pred, tf.argmax(y_, 1)), tf.float32))

# プレースホルダに与えるドロップアウトさせない率
train_prob = {prob_input: 1.0, prob_common: 1.0, prob_output: 0.9}
eval_prob = {prob_input: 1.0, prob_common: 1.0, prob_output: 1.0}

# プレースホルダに与える入力データとラベル
train_data = {x: mnist.train.images, y_: mnist.train.labels}
validation_data = {x: mnist.validation.images, y_: mnist.validation.labels}
test_data = {x: mnist.test.images, y_: mnist.test.labels}

# 学習の実行
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(5001):
        # trainデータはミニバッチにして学習させる
        batch = mnist.train.next_batch(32)
        inout = {x: batch[0], y_: batch[1]}
        if i % 500 == 0:
            # validationデータでevaluate
            train_accuracy = accuracy.eval(
                feed_dict={**validation_data, **eval_prob})
            print('step {:d}, accuracy {:d} ({:.2f})'.format(
                i, int(train_accuracy), train_accuracy / 5000))
        # 学習
        optimizer.run(feed_dict={**inout, **train_prob})

    # testデータでevaluate
    test_accuracy = accuracy.eval(feed_dict={**test_data, **eval_prob})
    print('test accuracy {:d} ({:.2f})'.format(
        int(test_accuracy), test_accuracy / 10000))
