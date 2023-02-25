import tensorflow as tf 

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.graph_options.rewrite_options.tensorflow_intra_op_parallelism = 1
config.graph_options.rewrite_options.tensorflow_inter_op_parallelism = 1
config.graph_options.rewrite_options.constant_folding = (
    tf.OptimizerOptions.OFF
)
config.graph_options.rewrite_options.arithmetic_optimization = (
    tf.OptimizerOptions.OFF
)
config.graph_options.rewrite_options.scoped_allocator_optimization = (
    tf.OptimizerOptions.ON_1
)
config.graph_options.rewrite_options.pin_to_host_optimization = (
    tf.OptimizerOptions.ON_1
)
config.graph_options.rewrite_options.auto_mixed_precision = (
    tf.OptimizerOptions.OFF
)
config.graph_options.rewrite_options.remapping = (
    tf.OptimizerOptions.OFF
)
config.graph_options.rewrite_options.layout_optimizer = (
    tf.OptimizerOptions.ON
)
config.graph_options.rewrite_options.min_graph_nodes = (
    tf.OptimizerOptions.OFF
)
config.graph_options.rewrite_options.memory_optimization = (
    tf.OptimizerOptions.OFF
)

config.graph_options.rewrite_options.trt_optimization = (
    tf.compat.v1.GraphRewriteSpec.OFF
)
