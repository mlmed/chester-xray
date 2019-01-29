import onnx
import tensorflow as tf
import sys
from os.path import dirname
sys.path.append(dirname("onnx-tensorflow/"))
import onnx_tf
from onnx_tf.backend import prepare
import numpy as np

# onnx_model = onnx.load("chest.onnx")  # load onnx model
# onnx.checker.check_model(onnx_model)

# from onnx import optimizer

# optimized_model = optimizer.optimize(onnx_model)
# tf_rep = onnx_tf.backend.prepare(optimized_model, strict=False)  # prepare tf representation
# tf_rep.export_graph("chest2.pb")  # export the model


def load_graph_base(frozen_graph_filename):

    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        new_input = tf.placeholder(tf.float32, [1,3,224,224], name="new_input")
        
        x,z = tf.import_graph_def(
                graph_def, 
                input_map={"0:0": new_input},
                name="prefix",
                producer_op_list=None,
              return_elements=[
                  "0:0",
                  "Sigmoid:0",
              ]
            )
        
    return graph, x, z

def load_graph(frozen_graph_filename):

    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        new_input = tf.placeholder(tf.float32, [1,3,224,224], name="new_input")
        
        @tf.RegisterGradient("CustomReluGrad")
        def _custom_relu_grad(op, grad):
            return tf.multiply(grad, tf.cast(tf.greater(op.outputs[0], 0.), tf.float32))
        @tf.RegisterGradient("CustomSigmoidGrad")
        def _custom_sigmoid_grad(op, grad):
            return tf.multiply(op.outputs[0], 1. - op.outputs[0])
        
        def unbroadcast(grad, shape):
            grad_shape = tf.shape(grad)
            num_dims_grad = tf.shape(grad_shape)
            num_dims_input = tf.shape(shape)
            grad = tf.reduce_sum(grad, axis=tf.range(num_dims_grad[0] - num_dims_input[0]))
            new_grad_shape = tf.shape(grad)
            mask = tf.logical_and(tf.equal(shape, 1), tf.not_equal(new_grad_shape, 1))
            indice = tf.boolean_mask(tf.range(num_dims_input[0]), mask)
            grad = tf.reduce_sum(grad, axis=indice, keepdims=True)
            return grad
            
        @tf.RegisterGradient("CustomAddGrad")
        def _custom_add_grad(op, grad):
            return tuple(unbroadcast(grad, tf.shape(input)) for input in op.inputs)
        
        @tf.RegisterGradient("CustomMulGrad")
        def _custom_mul_grad(op, grad):
            return unbroadcast(tf.multiply(grad, op.inputs[1]), tf.shape(op.inputs[0])), \
                    unbroadcast(tf.multiply(grad, op.inputs[0]), tf.shape(op.inputs[1]))
        
        @tf.RegisterGradient("CustomTransposeGrad")
        def _custom_transpose_grad(op, grad):
            value, perm = op.inputs[0], op.inputs[1]
            _, reverse_perm = tf.math.top_k(perm, k=tf.shape(perm)[0])
            return tf.transpose(grad, tf.reverse(reverse_perm, axis=[0])), None
        
        
        with graph.gradient_override_map({"Relu": "CustomReluGrad",
                                          "Sigmoid": "CustomSigmoidGrad",
                                          "Add": "CustomAddGrad",
                                          "Mul": "CustomMulGrad",
                                          "Transpose": "CustomTransposeGrad"}):
            
            x,z = tf.import_graph_def(
                graph_def, 
                input_map={"0:0": new_input},
                name="prefix",
                producer_op_list=None,
              return_elements=[
                  "0:0",
                  "Sigmoid:0",
              ]
            )
        
    return graph, x, z


print("===load chest2.pb")
graph, x, z = load_graph("chest2.pb")


print("===compute grad")
with graph.as_default():
    session = tf.Session(graph=graph)
    dx = tf.gradients(
        z[0],
        x,
        name='gradients'
    )
    print(dx)
    


print("===test grads work")
import scipy.misc, imageio, skimage.transform
img = imageio.imread("ChestXrays/00000120_000-Atelectasis.png")
a = skimage.transform.resize(img, (224, 224))
imgi = np.asarray([a,a,a]).reshape(1,3,224,224)

with graph.as_default():
    session = tf.Session(graph=graph)
    b, gb = session.run([z,dx], feed_dict={x:imgi})
    
print(gb)

    
    
print("===write chest2g.pb which has grads")
with graph.as_default():
    session = tf.Session(graph=graph)
    tf.io.write_graph(graph, as_text=False, logdir=".", name="chest2g.pb")



def load_graph2(frozen_graph_filename):
    
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        graph_def2 = tf.graph_util.extract_sub_graph(graph_def, ["prefix/Sigmoid", "gradients/prefix/Pad_grad/Slice_1"])
        
        x,z = tf.import_graph_def(
            graph_def2,
            name="",
            producer_op_list=None,
            return_elements=[
              "new_input:0",
              "prefix/Sigmoid:0"
            ]
        )
        
    return graph, x, z


print("===load chest2g.pb and extract subgraph")
graph, x, z = load_graph2("chest2g.pb")


print("===write chest2g.pb again")
with graph.as_default():
    session = tf.Session(graph=graph)
    tf.io.write_graph(graph, as_text=False, logdir=".", name="chest2g.pb")
    tf.io.write_graph(graph, as_text=True, logdir=".", name="chest2g.pbtxt")





































