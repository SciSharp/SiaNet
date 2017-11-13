![Build Status](https://travis-ci.org/deepakkumar1984/SiaNet.svg?branch=master)
[![Join the chat at https://gitter.im/sia-cog/SiaNet](https://badges.gitter.im/sia-cog/SiaNet.svg)](https://gitter.im/sia-cog/SiaNet?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

# A C# deep learning wrapper with CNTK backend

Developing a C# wrapper to help developer easily create and train deep neural network models. I am working on enhancing the interface to load data, build model, train and predict.

## Install using NuGet
GPU and CPU Version: [https://www.nuget.org/packages/SiaNet](https://www.nuget.org/packages/SiaNet)

For better performance on CPU please use CPU only version.
CPU Only Version: [https://www.nuget.org/packages/SiaNet.CPUOnly/](https://www.nuget.org/packages/SiaNet.CPUOnly/)

## Load dataset (Housing regression example)
```DataFrame frame = new DataFrame();```

```frame.LoadFromCsv(trainFile);```

```var xy = frame.SplitXY(14, new[] { 1, 13 });```

```traintest = xy.SplitTrainTest(0.25);```

## Load Sample Dataset (MNIST)
```Downloader.DownloadSample(SampleDataset.MNIST);```

```var samplePath = Downloader.GetSamplePath(SampleDataset.MNIST);```

```train = ImageDataGenerator.FlowFromText(samplePath.Train);```

```validation = ImageDataGenerator.FlowFromText(samplePath.Test);```

## Build Model
```model = new Sequential();```

```model.Add(new Dense(13, 12, OptActivations.ReLU));```

```model.Add(new Dense(13, OptActivations.ReLU));```

```model.Add(new Dense(1));```

## Build Convolution Layers
```model.Add(new Conv2D(Tuple.Create(imageDim[0], imageDim[1], imageDim[2]), 4, Tuple.Create(3, 3), Tuple.Create(2, 2), activation: OptActivations.None, weightInitializer: OptInitializers.Xavier, useBias: true, biasInitializer: OptInitializers.Ones));```

```model.Add(new MaxPool2D(Tuple.Create(3, 3)));```

```model.Add(new Conv2D(8, Tuple.Create(3, 3), Tuple.Create(2, 2), activation: OptActivations.None, weightInitializer: OptInitializers.Xavier));```

```model.Add(new MaxPool2D(Tuple.Create(3, 3)));```

```model.Add(new Dense(numClasses));```

## Configure Training callbacks
```model.OnEpochEnd += Model_OnEpochEnd;```

```model.OnTrainingEnd += Model_OnTrainingEnd;```

```model.OnBatchEnd += Model_OnBatchEnd;```

## Train Model
```model.Compile(OptOptimizers.Adam, OptLosses.MeanSquaredError, OptMetrics.MAE, Regulizers.RegL2(0.1));```
```model.Train(traintest.Train, 64, 200, traintest.Test);```

API Documentation: https://deepakkumar1984.github.io/SiaNet/

Examples Docs (More to add)

MNIST Training: [https://github.com/deepakkumar1984/SiaNet/wiki/Example---MNIST-Training](https://github.com/deepakkumar1984/SiaNet/wiki/Example---MNIST-Training)

