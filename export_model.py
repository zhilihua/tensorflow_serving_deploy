import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
import os

def save_model_to_serving(model, export_version, export_path='prod_models'):
    print(model.input, model.output)
    signature = tf.saved_model.signature_def_utils.predict_signature_def(
        inputs={'images': model.input}, outputs={'scores': model.output})
    export_path = os.path.join(
        tf.compat.as_bytes(export_path),
        tf.compat.as_bytes(str(export_version)))
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    builder.add_meta_graph_and_variables(
        sess=K.get_session(),
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'catdog_classification': signature,
        },
        main_op=tf.tables_initializer(),
        strip_default_attrs=True)
    builder.save()
model = load_model('train_model\savemodel_1fc256.h5')
save_model_to_serving(model, "001", "animal_serving")#bgru_serving表示转换后的模型会存储到该路径下