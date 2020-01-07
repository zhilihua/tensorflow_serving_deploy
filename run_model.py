from tensorflow_serving_client.protos import predict_pb2, prediction_service_pb2_grpc
import grpc
import tensorflow as tf
import time
import numpy as np
from PIL import Image
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications.vgg16 import preprocess_input

def init_request():
    print('正在连接Tensorflow Serving...')
    channel = grpc.insecure_channel('192.168.10.100:8500')
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = "catdog"
    # request.model_spec.signature_name = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY  # 'predict_images'
    request.model_spec.signature_name = "catdog_classification"
    print('完成连接Tensorflow Serving...')
    return request, stub

request, stub = init_request()

def run_model(img_224X224):
    img_proto = tf.contrib.util.make_tensor_proto(img_224X224, dtype=tf.float32)
    request.inputs["images"].ParseFromString(img_proto.SerializeToString())

    response = stub.Predict(request, 10.0)  #第二个参数是最大等待时间，因为这里是block模式访问的
    results = {}
    for key in response.outputs:
        tensor_proto = response.outputs[key]
        nd_array = tf.contrib.util.make_ndarray(tensor_proto)
        results[key] = nd_array
    return results

def Data_preprocessing(img, target_size):
    if img.size != target_size:
        img = img.resize(target_size)

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

if __name__ == '__main__':
    # 文件读取和处理
    img = Image.open('1.jpg')
    target_size = (224, 224)
    img = Data_preprocessing(img, target_size)
    print(img.shape)

    start_time = time.time()
    results = run_model(img)
    print("cost %ss to predict: " % (time.time() - start_time))
    # print(results["classes"])
    print(results["scores"])
