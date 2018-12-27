using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet
{
    public enum ActivationType
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
}
