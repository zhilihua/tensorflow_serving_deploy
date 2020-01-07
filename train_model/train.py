from tensorflow.python.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.python.keras.layers import Dense, Flatten, Dropout
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import os
import glob

def get_nb_files(directory):   #获取文件夹下的文件数
  if not os.path.exists(directory):
    return 0
  cnt = 0
  for r, dirs, files in os.walk(directory):
    for dr in dirs:
      cnt += len(glob.glob(os.path.join(r, dr + "/*")))
  return cnt

def TrainModel(path, train_dir, val_dir, batch_size,
               epochs, out_nums, nb_train_samples,
               nb_val_samples, img_width=256, img_height=256, freeze=13):
    #生成训练和验证数据
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=40,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        # shear_range=0.2,
        # zoom_range=0.2,
        horizontal_flip=True, )  # 训练数据预处理器，随机水平翻转
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)  # 测试数据预处理器
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=(img_width, img_height),
                                                        batch_size=batch_size,
                                                        # class_mode='binary'
                                                        )  # 训练数据生成器
    validation_generator = test_datagen.flow_from_directory(val_dir,
                                                            target_size=(img_width, img_height),
                                                            batch_size=batch_size,
                                                            # class_mode='binary',
                                                            shuffle=True)  # 验证数据生成器

    base_model = VGG16(weights=path, include_top=False,     #加载迁移学习模型
                        input_shape=(img_width, img_height, 3))

    for ix, layers in enumerate(base_model.layers):
        if ix < freeze:    #冻结指定层
            layers.trainable = False
        # layers.trainable = False   #冻结指定层数层

    #添加新的层用于训练
    model = Flatten()(base_model.output)
    model = Dense(256, activation='relu', name='fc1')(model)
    model = Dropout(0.5, name='dropout1')(model)
    #=========================新加一层全连接=======================
    # model = Dense(64, activation='relu', name='fc2')(model)
    # model = Dropout(0.5, name='dropout2')(model)
    #==============================================================
    model = Dense(out_nums, activation='softmax')(model)
    # model = Dense(out_nums, activation='sigmoid')(model)
    model_final = Model(inputs=base_model.input, outputs=model)
    model_final.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.0001, momentum=0.9),
                  metrics=['accuracy'])
    # model_final.compile(loss='binary_crossentropy',
    #                     optimizer=SGD(lr=0.0001, momentum=0.9),
    #                     metrics=['accuracy'])
    print(model_final.summary())
    callbacks = [
        EarlyStopping(patience=2, verbose=1),
        ModelCheckpoint('savemodel_1fc256.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        # ModelCheckpoint('savemodel_1fc256_3conv_binary.h5', verbose=1, save_best_only=False, mode='max')
    ]

    # 训练&评估
    model_final.fit_generator(train_generator, steps_per_epoch=nb_train_samples // batch_size, epochs=epochs,
                        validation_data=validation_generator, validation_steps=nb_val_samples // batch_size,
                        callbacks=callbacks, initial_epoch=0)  # 每轮一行输出结果

if __name__ == '__main__':
    path = "model/model_vgg.h5"
    img_width, img_height = 224, 224  # 图片高宽
    batch_size = 16  # 批量大小
    epochs = 20  # 迭代次数
    train_data_dir = 'dataset/train'  # 训练集目录
    validation_data_dir = 'dataset/validation'  # 测试集目录
    OUT_CATEGORIES = 2  # 分类数
    freeze = 18 #冻结的卷积层

    Tcnt = get_nb_files(train_data_dir)  #得到训练集的数量
    Vcnt = get_nb_files(validation_data_dir)  #得到验证集的数量

    TrainModel(path=path,
               train_dir=train_data_dir,
               val_dir=validation_data_dir,
               batch_size=batch_size,
               epochs=epochs,
               out_nums=OUT_CATEGORIES,
               nb_train_samples=Tcnt,
               nb_val_samples=Vcnt,
               img_width=img_width,
               img_height=img_height,
               freeze=freeze
               )