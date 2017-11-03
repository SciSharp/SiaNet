![Build Status](https://travis-ci.org/deepakkumar1984/SiaNet.svg?branch=master)
[![Join the chat at https://gitter.im/sia-cog/SiaNet](https://badges.gitter.im/sia-cog/SiaNet.svg)](https://gitter.im/sia-cog/SiaNet?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

# A CSharp deep learning wrapper with CNTK backend

Developing a C# wrapper to help developer easily create and train deep neural network models. I am working on enhancing the interface to load data, build model, train and predict.

## Load dataset (Housing regression example)

DataFrame frame = new DataFrame();

frame.LoadFromCsv(trainFile);

var xy = frame.SplitXY(14, new[] { 1, 13 });

traintest = xy.SplitTrainTest(0.25);


## Build Model
model = new Sequential();

model.Add(new Dense(13, 12, OptActivations.ReLU));

model.Add(new Dense(13, OptActivations.ReLU));

model.Add(new Dense(1));

## Configure Training Events

model.OnEpochEnd += Model_OnEpochEnd;

model.OnTrainingEnd += Model_OnTrainingEnd;

## Train Model

model.Compile(OptOptimizers.Adam, OptLosses.MeanSquaredError, OptMetrics.MAE, Regulizers.RegL2(0.1));

model.Train(traintest.Train, 64, 200, traintest.Test);




