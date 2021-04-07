import tensorflow as tf
from src.dataloaders import data_loader
import logging
import matplotlib.pyplot as plt

def validate_test(path,train_dataset, test_dataset):
    logging.info("=" * 40)
    model = tf.keras.models.load_model(path)
    logging.info("Validating model at ")
    logging.info(path)
    dot_img_file = root_path + "/src/test_images_new/" + path.split("/")[-1] + ".png"
    tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
    model.summary()

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    optimizer = "SGD"

    #for layer in model.layers:
    #    print(layer.get_config())
    #layer_names = ["dense_1","dense_30"]
    #for name in layer_names:
    #    print(model.get_layer(name).get_config())

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
    results = model.evaluate(test_dataset,verbose=1)
    print(results)
    model.evaluate(val_dataset, verbose=1)
    history = model.fit(train_dataset,epochs=120,verbose=1,validation_data=val_dataset)
    plt.plot(history.history['sparse_categorical_accuracy'])
    plt.plot(history.history['val_sparse_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.show()
    plt.title("accuracy graph for blueprint" + path.split("/")[-1])
    plt.savefig(root_path + "/src/test_images_new/accuracy_"+ path.split("/")[-1] + ".png")
    plt.close()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.title("loss graph for blueprint" + path.split("/")[-1])
    plt.savefig(root_path + "/src/test_images_new/loss_"+ path.split("/")[-1]+ ".png")
    plt.close()

    results = model.evaluate(test_dataset, verbose=1)
    logging.info(results)
    logging.info("="*40)
    return results

if __name__=="__main__":
    
    root_path = "/home/vikas/workspace/MasterThesis/coderepo/backups/Super-CoDeepNeat"
    path = root_path + "/saved_models/modelblueprint-61083719_07_52_27_03_2021"
    train_dataset, val_dataset, test_dataset = data_loader.load_cifar10()
    results = validate_test(path, train_dataset, test_dataset)
    print(results)
