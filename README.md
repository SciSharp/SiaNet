[![Build status](https://dev.azure.com/deepakkumarb/SIA/_apis/build/status/SiaNet%20Beta%200.4.1)](https://dev.azure.com/deepakkumarb/SIA/_build/latest?definitionId=4)
![Build Status](https://travis-ci.org/deepakkumar1984/SiaNet.svg?branch=master)
[![Backers on Open Collective](https://opencollective.com/sianet/backers/badge.svg)](#backers) [![Sponsors on Open Collective](https://opencollective.com/sianet/sponsors/badge.svg)](#sponsors) 

[<img src="https://img.shields.io/badge/slack-@siadroid/sianet-green.svg?logo=slack">](https://siadroid.slack.com/messages/CGL4QULPM)

Trello is used to track SiaNet devlopment activities. You are welcome to watch any task and track progress. Suggestion will be put on the wishlist and then will be planned out for development

https://trello.com/b/bLbgQLgy/sianet-development



# A C# deep learning library

Developing a C# wrapper to help developer easily create and train deep neural network models.

* Easy to use library, just focus on research
* Multiple backend - CNTK, TensorFlow, MxNet, ArrayFire, TensorSharp
* CUDA/ OpenCL support for some of the backends
* Light weight libray, built with .NET standard 2.0
* Code well structured, easy to extend if you would like to extend with new layer, loss, metrics, optimizers, constraints, regularizer

# A Basic example
The below is a classification example with Titanic dataset. Able to reach 75% accuracy within 10 epoch. 
```
//Setup Engine. If using TensorSharp then pass SiaNet.Backend.TensorSharp.SiaNetBackend.Instance. 
//Once other backend is ready you will be able to use CNTK, TensorFlow and MxNet as well.
Global.UseEngine(SiaNet.Backend.ArrayFire.SiaNetBackend.Instance, DeviceType.CPU);

var dataset = LoadTrain(); //Load train data
var test = LoadTest(); //Load test data

var (train, val) = dataset.Split(0.25);

//Build model
var model = new Sequential();
model.EpochEnd += Model_EpochEnd;
model.Add(new Dense(128, ActivationType.ReLU));
model.Add(new Dense(64, ActivationType.ReLU));
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

Complete Code: https://github.com/SciSharp/SiaNet/blob/master/Examples/BasicClassificationWithTitanicDataset/Program.cs

More examples: https://github.com/SciSharp/SiaNet/blob/master/Examples

# API Docs
https://scisharp.github.io/SiaNet/

# Contribution
Any help is welcome!!!


