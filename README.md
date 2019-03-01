`![Build Status](https://travis-ci.org/deepakkumar1984/SiaNet.svg?branch=master)
[![Backers on Open Collective](https://opencollective.com/sianet/backers/badge.svg)](#backers) [![Sponsors on Open Collective](https://opencollective.com/sianet/sponsors/badge.svg)](#sponsors) 

[<img src="https://img.shields.io/badge/slack-@siadroid/sianet-green.svg?logo=slack">](https://siadroid.slack.com/messages/CGL4QULPM)

Trello is used to track SiaNet devlopment activities. You are welcome to watch any task and track progress. Suggestion will be put on the wishlist and then will be planned out for development

https://trello.com/b/bLbgQLgy/sianet-development



# A C# deep learning library

Developing a C# wrapper to help developer easily create and train deep neural network models.

* Easy to use library, just focus on research
* Multiple backend - ArrayFire (In Progress), TensorSharp (In Progress), CNTK (Not Started), TensorFlow (Not Started), MxNet (Not Started)
* CUDA/ OpenCL support for some of the backends
* Light weight libray, built with .NET standard 2.0
* Code well structured, easy to extend if you would like to extend with new layer, loss, metrics, optimizers, constraints, regularizer

# Current Work In Progress
I am currently working on ArrayFire and TensorSharp backend. If using ArrayFIre please download the windows installer from https://arrayfire.com/download/

For TensorSharp you don't have to install anything. To use any CUDA version > 8.0 and build the project with the specific cuda installed in your machine, please change the "Conditional Compilation Property" in the ManagedCuda project properties -> Build.

Following are the combination: 

{WIN/LINUX},{CUDA80/CUDA90/CUDA91/CUDA92/CUDA100},{CUDNN5/CUDNN7/NA}

Ex: WIN,CUDA100,CUDNN7 - For Cuda 10 with CuDNN7 on Windows

Ex: LINUX,CUDA90,CUDNN5 - For Cuda 9 with CuDNN5 on Linux

Ex: WIN,CUDA92,NA - For Cuda 10 with No CuDNN installed on Windows

# A Basic example
The below is a classification example with Titanic dataset. Able to reach 75% accuracy within 10 epoch. 
```
//Setup Engine. If using TensorSharp then pass SiaNet.Backend.TensorSharp.SiaNetBackend.Instance. Once other backend is developed you 
//will be able to use CNTK, TensorFlow and MxNet as well.
Global.UseEngine(SiaNet.Backend.ArrayFire.SiaNetBackend.Instance, DeviceType.CPU);

var dataset = LoadTrain(); //Load train data
var test = LoadTest(); //Load test data

var (train, val) = dataset.Split(0.25);

//Build model
var model = new Sequential();
model.EpochEnd += Model_EpochEnd;
model.Add(new Dense(16, ActivationType.ReLU));
model.Add(new Dense(8, ActivationType.ReLU));
model.Add(new Dense(1, ActivationType.Sigmoid));

//Compile with Optimizer, Loss and Metric
model.Compile(OptimizerType.Adam, LossType.BinaryCrossEntropy, MetricType.BinaryAccurary);

// Train for 100 epoch with batch size of 32
model.Train(train, 100, 32, val);

var predictions = model.Predict(test);
predictions.Print();
```
### Training Result

![Figure 1-1](https://i.ibb.co/KG87pv4/Titanic-1.png "Figure 1-1")

Complete Code: https://github.com/deepakkumar1984/SiaNet/blob/master/Examples/BasicClassificationWithTitanicDataset/Program.cs

More examples: https://github.com/deepakkumar1984/SiaNet/blob/master/Examples

# Contribution
Any help is welcome!!!


