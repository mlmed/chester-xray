import sys
import pkg_resources


def convert_onnx2tf_pb(onnx_file="chest.onnx", pb_file="chest.pb"):
    
    for package in ["onnx","onnx_tf", "tensorflow"]:
        print(pkg_resources.get_distribution(package))
    import onnx, onnx.optimizer
    import tensorflow as tf
    import onnx_tf

    # load onnx model
    onnx_model = onnx.load(onnx_file)
    onnx_model.ir_version = 4
    onnx.checker.check_model(onnx_model)

    optimized_model = onnx_model#onnx.optimizer.optimize(onnx_model)
    
    tf_rep = onnx_tf.backend.prepare(optimized_model, strict=False) 
    tf_rep.export_graph(pb_file)
    
def convert_tf_pb2savedmodel(pb_file="chest.pb", savedmodel_folder="chest-savedmodel", input_node=None, output_node=None):
    # from here: https://stackoverflow.com/questions/44329185/convert-a-graph-proto-pb-pbtxt-to-a-savedmodel-for-use-in-tensorflow-serving-o
    for package in ["tensorflow"]:
        print(pkg_resources.get_distribution(package))
    import tensorflow as tf
    from tensorflow.python.saved_model import signature_constants
    from tensorflow.python.saved_model import tag_constants

    builder = tf.saved_model.builder.SavedModelBuilder(savedmodel_folder)

    with tf.gfile.GFile(pb_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    sigs = {}

    with tf.Session(graph=tf.Graph()) as sess:
        # name="" is important to ensure we don't get spurious prefixing
        tf.import_graph_def(graph_def, name="")
        g = tf.get_default_graph()
        gdef = g.as_graph_def()
        if input_node == None:
            maybe_inputs = [(n.name, n.op) for n in gdef.node if n.op in ('Placeholder')]
            print("Possible inputs:", maybe_inputs)
            input_node = "{}:0".format(maybe_inputs[0][0])
            print("Selecting first element in list: {}".format(input_node))
        if output_node == None:   
            maybe_outputs = [(n.name, n.op) for n in gdef.node if n.op in ('Softmax','Sigmoid')]
            maybe_outputs.append((gdef.node[-1].name, gdef.node[-1].op))
            print("Possible outputs:", maybe_outputs)
            output_node = "{}:0".format(maybe_outputs[0][0])
            print("Selecting first element in list: {}".format(output_node))

        inp = g.get_tensor_by_name(input_node)
        out = g.get_tensor_by_name(output_node)

        sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
            tf.saved_model.signature_def_utils.predict_signature_def(
                {"in": inp}, {"out": out})

        builder.add_meta_graph_and_variables(sess,
                                             [tag_constants.SERVING],
                                             signature_def_map=sigs)

    builder.save()
    return sigs
    
def convert_tf2tfjs(savedmodel_folder="chest-test", tfjs_folder="chest-tfjs"):
    pass



