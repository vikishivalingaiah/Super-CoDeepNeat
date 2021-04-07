# Evolving Deep Neural Networks with Parameter Sharing

The repo contains the implementaion of CoDeepNeat with parameter sharing. 


Folder Structure:
```
.
├── logs
│   ├── evolution_log_15_20_11_02_2021.log
│   └── evolution_log_16_39_20_03_2021.log
├── README.md
├── requirements.txt
├── saved_models
│   ├── modelblueprint-05267813_19_27_17_03_2021
│   │   ├── assets
│   │   ├── saved_model.pb
│   │   └── variables
│   │       ├── variables.data-00000-of-00001
│   │       └── variables.index
│   └── modelblueprint-61083719_07_52_27_03_2021
│       ├── assets
│       ├── saved_model.pb
│       └── variables
│           ├── variables.data-00000-of-00001
│           └── variables.index
├── src
│   ├── configuration
│   │   ├── __init__.py
│   │   ├── nnlayers.py
│   │   └── __pycache__
│   │       ├── __init__.cpython-36.pyc
│   │       └── nnlayers.cpython-36.pyc
│   ├── dataloaders
│   │   ├── data_loader.py
│   │   ├── __init__.py
│   │   └── __pycache__
│   │       ├── data_loader.cpython-36.pyc
│   │       └── __init__.cpython-36.pyc
│   ├── encoding
│   │   ├── blueprint.py
│   │   ├── component.py
│   │   ├── DAGEncoder.py
│   │   ├── __init__.py
│   │   ├── layer.py
│   │   ├── module.py
│   │   ├── __pycache__
│   │   │   ├── blueprint.cpython-36.pyc
│   │   │   ├── component.cpython-36.pyc
│   │   │   ├── DAGEncoder.cpython-36.pyc
│   │   │   ├── __init__.cpython-36.pyc
│   │   │   ├── layer.cpython-36.pyc
│   │   │   ├── module.cpython-36.pyc
│   │   │   └── supernet.cpython-36.pyc
│   │   └── supernet.py
│   ├── evolution
│   │   ├── evolution.py
│   │   ├── historical_marker.py
│   │   ├── historical_supernets.py
│   │   ├── initialization.py
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── evolution.cpython-36.pyc
│   │   │   ├── historical_marker.cpython-36.pyc
│   │   │   ├── historical_supernets.cpython-36.pyc
│   │   │   ├── __init__.cpython-36.pyc
│   │   │   ├── initialization.cpython-36.pyc
│   │   │   ├── Operators.cpython-36.pyc
│   │   │   └── species.cpython-36.pyc
│   │   └── species.py
│   ├── __init__.py
│   ├── main
│   │   ├── __init__.py
│   │   └── mainDAG.py
│   ├── __pycache__
│   │   └── __init__.cpython-36.pyc
│   ├── test_images
│   ├── test_images_new
│   │   ├── accuracy_modelblueprint-61083719_08_56_27_03_2021.png
│   │   ├── loss_modelblueprint-61083719_08_56_27_03_2021.png
│   │   ├── modelblueprint-61083719_07_52_27_03_2021.png
│   │   └── modelblueprint-61083719_08_56_27_03_2021.png
│   ├── training
│   │   ├── custom_layers.py
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── __pycache__
│   │   │   ├── custom_layers.cpython-36.pyc
│   │   │   ├── __init__.cpython-36.pyc
│   │   │   ├── models.cpython-36.pyc
│   │   │   └── trainer.cpython-36.pyc
│   │   ├── trainer.py
│   │   └── validate_model.py
│   └── utils
│       ├── __init__.py
│       ├── pdfsummary.py
│       └── __pycache__
│           ├── __init__.cpython-36.pyc
│           └── pdfsummary.cpython-36.pyc
└── summary
    ├── cifar10_conv_summary_bkp.pdf
    ├── cifar10_conv_summary.pdf
    └── conv_summary.pdf

```

Usage:

The src/main/mainDAG.py file has the functions to evolve CNNs for CIFAR-10 and MNIST. 

Running this will gneerate images in test_images folder, saves models in saved_models folder. 

After the run, The best model can be obtained from logs. The best model can be furtehr trained using src/training/validat_model.py. Running this will generate training,validation loss and accuracy curves in test_images_new folder.

The saved_models folder contains models for CIFAR-10 searched with full dataset and reduced dataset sizes. 


The configuration for supernets, evolution parameters can be modified from src/configuŕation/nnlayers.py. 
The datasets size can be modified form src/dataloaders/data_loader.py.

The summary folder contains PDF which contains results of speciation and generation wise fitness, blueprint graphs etc. 
