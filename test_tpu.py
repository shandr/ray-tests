import ray
import tensorflow as tf


@ray.remote
def try_tpu():
    print("Tensorflow version " + tf.__version__)

    import os

    print(f'ENV: {os.environ}')

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=os.environ['KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS']) # '$(KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS)'
    print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])

    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)

    @tf.function
    def add_fn(x,y):
        z = x + y
        return z

    x = tf.constant(1.)
    y = tf.constant(1.)
    z = strategy.run(add_fn, args=(x,y))
    print(z)


ray.init()


print(f"RESULT IS {ray.get(try_tpu.remote())}")
