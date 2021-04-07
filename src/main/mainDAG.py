from src.encoding.DAGEncoder import DAGEncoder
from src.evolution.initialization import  Initialization
from src.evolution.historical_supernets import HistoricalSupernetMap
from src.configuration import nnlayers
import networkx as nx
import matplotlib.pyplot as plt
from src.dataloaders import data_loader
from src.training.models import ComponentModel,ModuleModel,SupernetModel
import tensorflow as tf
from random import choice
from src.evolution.historical_marker import HistoricalMarker
from src.training.trainer import Trainer
from src.evolution.species import Species
from src.evolution.evolution import Evolution
import logging
import atexit
from datetime import datetime
import sys
from matplotlib.backends.backend_pdf import PdfPages


def evolve_with_supernet():
    train_dataset, val_dataset = data_loader.load_iris()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    optimizer = "SGD"
    input_layer = tf.keras.layers.Input(shape=(4))
    output_layer = tf.keras.layers.Dense(3)
    pdf_pages = PdfPages(root_path + "/summary/" + "summary.pdf")
    trainer = Trainer(train_dataset, val_dataset,val_dataset, input_layer, output_layer, optimizer, loss_fn, metrics, 1,root_path)
    evolution = Evolution(nnlayers, trainer, pdfpages=pdf_pages)

    evolution.evolve()
    pdf_pages.close()


def evolve_with_supernet_mnist():
    train_dataset, val_dataset, test_dataset = data_loader.load_mnist()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    optimizer = "SGD"
    input_layer = tf.keras.layers.Input(shape=(28,28,1))
    output_model = tf.keras.Sequential()
    output_model.add(tf.keras.layers.Flatten())
    output_model.add(tf.keras.layers.Dense(256))
    output_model.add(tf.keras.layers.Dense(10))
    pdf_pages = PdfPages(root_path + "/summary/" + "conv_summary.pdf")
    trainer = Trainer(train_dataset, val_dataset, test_dataset, input_layer, output_model, optimizer, loss_fn, metrics, 1,root_path)
    evolution = Evolution(nnlayers, trainer, pdfpages=pdf_pages,root_path=root_path)

    evolution.evolve()
    pdf_pages.close()



def evolve_with_supernet_cifar10():
    
    def pp_close():
        pdf_pages.close()
        print("Program terminated,closed pdf")
    atexit.register(pp_close)

    train_dataset, val_dataset, test_dataset = data_loader.load_cifar10()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    optimizer =  tf.keras.optimizers.SGD(lr=0.001, momentum=0.9)
    input_layer = tf.keras.layers.Input(shape=(32,32,3))
    output_model = tf.keras.Sequential()
    output_model.add(tf.keras.layers.Flatten())
    output_model.add(tf.keras.layers.Dense(256))
    output_model.add(tf.keras.layers.Dense(10))
    pdf_pages = PdfPages(root_path + "/summary/" + "cifar10_conv_summary.pdf")
    trainer = Trainer(train_dataset, val_dataset,test_dataset, input_layer, output_model, optimizer, loss_fn, metrics, 1,root_path)
    evolution = Evolution(nnlayers, trainer, pdfpages=pdf_pages,root_path=root_path)

    evolution.evolve()
    pdf_pages.close()














if __name__=="__main__":
    logfilename = datetime.now().strftime('evolution_log_%H_%M_%d_%m_%Y.log')
    logging.basicConfig(filename="../../logs/" + logfilename, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    root_path = "/home/student/coceenna"
    #evolve_with_supernet()
    evolve_with_supernet_mnist()
    #evolve_with_supernet_cifar10()

   