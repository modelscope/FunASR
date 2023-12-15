import numpy as np
np.set_printoptions(threshold=np.inf)
import logging

def load_ckpt(checkpoint_path):
	import tensorflow as tf
	if tf.__version__.startswith('2'):
		import tensorflow.compat.v1 as tf
		tf.disable_v2_behavior()
		reader = tf.compat.v1.train.NewCheckpointReader(checkpoint_path)
	else:
		from tensorflow.python import pywrap_tensorflow
		reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
	var_to_shape_map = reader.get_variable_to_shape_map()

	var_dict = dict()
	for var_name in sorted(var_to_shape_map):
		if "Adam" in var_name:
			continue
		tensor = reader.get_tensor(var_name)
		# print("in ckpt: {}, {}".format(var_name, tensor.shape))
		# print(tensor)
		var_dict[var_name] = tensor

	return var_dict



def load_tf_pb_dict(pb_model):
	import tensorflow as tf
	if tf.__version__.startswith('2'):
		import tensorflow.compat.v1 as tf
		tf.disable_v2_behavior()
		# import tensorflow_addons as tfa
		# from tensorflow_addons.seq2seq.python.ops import beam_search_ops
	else:
		from tensorflow.contrib.seq2seq.python.ops import beam_search_ops
	from tensorflow.python.ops import lookup_ops as lookup
	from tensorflow.python.framework import tensor_util
	from tensorflow.python.platform import gfile
	
	sess = tf.Session()
	with gfile.FastGFile(pb_model, 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		sess.graph.as_default()
		tf.import_graph_def(graph_def, name='')
	
	var_dict = dict()
	for node in sess.graph_def.node:
		if node.op == 'Const':
			value = tensor_util.MakeNdarray(node.attr['value'].tensor)
			if len(value.shape) >= 1:
				var_dict[node.name] = value
	return var_dict

def load_tf_dict(pb_model):
	if "model.ckpt-" in pb_model:
		var_dict = load_ckpt(pb_model)
	else:
		var_dict = load_tf_pb_dict(pb_model)
	return var_dict
