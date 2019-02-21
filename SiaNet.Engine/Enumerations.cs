using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Engine
{
    public enum DataType
    {
        Float32,
        Float64,
        Int8,
        Int32
    }

    public enum ActType
    {
        Linear,
        ReLU,
        Sigmoid,
        Tanh,
        Elu,
        Exp,
        HargSigmoid,
        LeakyReLU,
        PReLU,
        RReLU,
        SeLU,
        Softmax,
        Softplus,
        SoftSign
    }

    public enum OptimizerType
    {
        SGD,
        RMSprop,
        Adagrad,
        Adadelta,
        Adamax,
        Adam
    }

    public enum LossType
    {
        MeanSquaredError,
        MeanAbsoluteError,
        MeanAbsolutePercentageError,
        MeanAbsoluteLogError,
        SquaredHinge,
        Hinge,
        BinaryCrossEntropy,
        CategorialCrossEntropy,
        CTC,
        KullbackLeiblerDivergence,
        Logcosh,
        Poisson,
        CosineProximity
    }
    public enum MetricType
    {
        None,
        Accuracy,
        BinaryAccurary,
        MSE,
        MAE,
        MAPE,
        MSLE
    }

    public enum PoolingPoolType
    {
        Max,
        Avg
    }

    public enum PaddingType
    {
        Valid,
        Same,
        Full
   }

    public enum DeviceType
    {
        Default,
        CPU,
        CUDA,
        OpenCL
    }

    public enum SiaNetBackend
    {
        TensorSharp,
        ArrayFire,
        CNTK,
        Tensorflow,
        MxNet
    }
}
