import tensorflow as tf

g = tf.Graph()

with g.as_default() as g:
    tf.train.import_meta_graph('./checkpoint/haoyao_style_gan/AnimeGAN.model-60.meta')

with tf.Session(graph=g) as sess:
    file_writer = tf.summary.FileWriter(logdir='./checkpoint/AnimeGAN_haoyao', graph=g)