![Build Status](https://travis-ci.org/deepakkumar1984/SiaNet.svg?branch=master)
[![Backers on Open Collective](https://opencollective.com/sianet/backers/badge.svg)](#backers) [![Sponsors on Open Collective](https://opencollective.com/sianet/sponsors/badge.svg)](#sponsors) [![Join the chat at https://gitter.im/sia-cog/SiaNet](https://badges.gitter.im/sia-cog/SiaNet.svg)](https://gitter.im/sia-cog/SiaNet?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

# A C# deep learning library

Developing a C# wrapper to help developer easily create and train deep neural network models.

* Easy to use library, just focus on research
* CUDA supported, no seperate package for each cuda version. supported so far: 8, 9.0, 9.1, 9.2, 10
* Light weight libray, built with .NET standard 2.0
* Code well structured, easy to extend if you would like to extend with new layer, loss, metrics, optimizers, constraints, regularizer


# A Basic example
The below is a classification example with Titainc dataset. Able to reach 75% accuravy in 10 epoch. 
```
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

// Train for 100 epoch with batch size of 2
model.Train(train, 100, 32, val);

var predictions = model.Predict(test);
predictions.Print();
```
### Training Result

![Figure 1-1](https://i.ibb.co/KG87pv4/Titanic-1.png "Figure 1-1")

Complete Code: https://github.com/deepakkumar1984/SiaNet/blob/master/Examples/BasicClassificationWithTitanicDataset/Program.cs

More examples: https://github.com/deepakkumar1984/SiaNet/blob/master/Examples
