using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet.Common
{
    /// <summary>
    /// List of supported layers names
    /// </summary>
    public class OptLayers
    {
        public const string Dense = "DENSE";
        public const string Activation = "ACTIVATION";
        public const string Dropout = "DROPOUT";
        public const string BatchNorm = "BATCHNORM";
        public const string Embedding = "EMBEDDING";
        public const string Conv1D = "CONV1D";
        public const string Conv2D = "CONV2D";
        public const string Conv3D = "CONV3D";
        public const string MaxPool1D = "MAXPOOL1D";
        public const string MaxPool2D = "MAXPOOL2D";
        public const string MaxPool3D = "MAXPOOL3D";
        public const string AvgPool1D = "AVGPOOL1D";
        public const string AvgPool2D = "AVGPOOL2D";
        public const string AvgPool3D = "AVGPOOL3D";
        public const string GlobalMaxPool1D = "GLOBALMAXPOOL1D";
        public const string GlobalMaxPool2D = "GLOBALMAXPOOL2D";
        public const string GlobalMaxPool3D = "GLOBALMAXPOOL3D";
        public const string GlobalAvgPool1D = "GLOBALAVGPOOL1D";
        public const string GlobalAvgPool2D = "GLOBALAVGPOOL2D";
        public const string GlobalAvgPool3D = "GLOBALAVGPOOL3D";
    }

    /// <summary>
    /// List of supported activation function names
    /// </summary>
    public class OptActivations
    {
        public const string None = "none";
        public const string ReLU = "relu";
        public const string LeakyReLU = "leakyrelu";
        public const string Sigmoid = "sigmoid";
        public const string Tanh = "tanh";
        public const string Softmax = "softmax";
        public const string Softplus = "softplus";
        public const string ELU = "elu";
    }

    /// <summary>
    /// List of supported initialize function names
    /// </summary>
    public class OptInitializers
    {
        public const string None = "none";
        public const string Uniform = "uniform";
        public const string Normal = "normal";
        public const string TruncatedNormal = "truncated_normal";
        public const string Zeros = "zeros";
        public const string Ones = "ones";
        public const string Constant = "constant";
        public const string Xavier = "xavier";
        public const string GlorotNormal = "glorot_normal";
        public const string GlorotUniform = "glorot_uniform";
        public const string HeNormal = "he_normal";
        public const string HeUniform = "he_uniform";
    }

    /// <summary>
    /// List of regulizer function name
    /// </summary>
    public class OptRegulizers
    {
        public const string None = "none";
        public const string L1 = "l1";
        public const string L2 = "l2";
        public const string L1L2 = "l1l2";
    }

    /// <summary>
    /// List of supported optimizer function name
    /// </summary>
    public class OptOptimizers
    {
        public const string SGD = "sgd";
        public const string MomentumSGD = "momentum_sgd";
        public const string RMSProp = "rmsprop";
        public const string Adam = "adam";
        public const string Adamax = "adamax";
        public const string AdaGrad = "adagrad";
        public const string AdaDelta = "adadelta";
    }

    /// <summary>
    /// List of supported metric name
    /// </summary>
    public class OptMetrics
    {
        public const string Accuracy = "acc";
        public const string TopKAccuracy = "top_k_acc";
        public const string MSE = "mse";
        public const string MAE = "mae";
        public const string MAPE = "mape";
        public const string MSLE = "msle";
    }

    /// <summary>
    /// List of supported loss function names
    /// </summary>
    public class OptLosses
    {
        public const string MeanSquaredError = "mean_squared_error";
        public const string MeanAbsoluteError = "mean_absolute_error";
        public const string MeanAbsolutePercentageError = "mean_absolute_percentage_error";
        public const string MeanSquaredLogError = "mean_squared_logarithmic_error";
        public const string CrossEntropy = "cross_entropy";
        public const string SparseCrossEntropy = "sparse_cross_entropy";
        public const string BinaryCrossEntropy = "binary_cross_entropy";
        public const string CTC = "ctc";
        public const string KullbackLeiblerDivergence = "kullback_leibler_divergence";
        public const string Poisson = "poisson";
    }

    /// <summary>
    /// Image Net pretrained model list
    /// </summary>
    public enum ImageNetModel
    {
        /// <summary>
        /// The Alexnet model
        /// </summary>
        AlexNet,

        /// <summary>
        /// The inception v3 model
        /// </summary>
        InceptionV3,

        /// <summary>
        /// The ResNet18 model
        /// </summary>
        ResNet18,

        /// <summary>
        /// The ResNet34 model
        /// </summary>
        ResNet34,

        /// <summary>
        /// The ResNet50 model
        /// </summary>
        ResNet50,

        /// <summary>
        /// The ResNet101 model
        /// </summary>
        ResNet101,

        /// <summary>
        /// The ResNet152 model
        /// </summary>
        ResNet152,

        /// <summary>
        /// The VGG16 model
        /// </summary>
        VGG16,

        /// <summary>
        /// The VGG19
        /// </summary>
        VGG19
    }

    /// <summary>
    /// Cifar-10 pretrained model list
    /// </summary>
    public enum Cifar10Model
    {
        /// <summary>
        /// Cifar-10 ResNet20 model
        /// </summary>
        ResNet20,

        /// <summary>
        /// Cifar-10 ResNet110 model
        /// </summary>
        ResNet110
    }

    /// <summary>
    /// Fast-RCNN pretrained model list
    /// </summary>
    public enum FastRCNNModel
    {
        /// <summary>
        /// Fast-RCNN Grocery100 model
        /// </summary>
        Grocery100,

        /// <summary>
        /// Fast-RCNN Pascal model
        /// </summary>
        Pascal
    }

    /// <summary>
    /// Pre trained model path to download from hosted sites
    /// </summary>
    public class PreTrainedModelPath
    {
        /// <summary>
        /// Path to ImageNet model files for various models like AlexNet, ResNet, VGG etc..
        /// </summary>
        public class ImageNetPath
        {
            public const string AlexNet = "https://www.cntk.ai/Models/CNTK_Pretrained/AlexNet_ImageNet_CNTK.model";
            public const string InceptionV3 = "https://www.cntk.ai/Models/CNTK_Pretrained/InceptionV3_ImageNet_CNTK.model";
            public const string ResNet18 = "https://www.cntk.ai/Models/CNTK_Pretrained/ResNet18_ImageNet_CNTK.model";
            public const string ResNet34 = "https://www.cntk.ai/Models/CNTK_Pretrained/ResNet34_ImageNet_CNTK.model";
            public const string ResNet50 = "https://www.cntk.ai/Models/CNTK_Pretrained/ResNet50_ImageNet_CNTK.model";
            public const string ResNet101 = "https://www.cntk.ai/Models/Caffe_Converted/ResNet101_ImageNet_Caffe.model";
            public const string ResNet152 = "https://www.cntk.ai/Models/Caffe_Converted/ResNet152_ImageNet_Caffe.model";
            public const string VGG16 = "https://www.cntk.ai/Models/Caffe_Converted/VGG16_ImageNet_Caffe.model";
            public const string VGG19 = "https://www.cntk.ai/Models/Caffe_Converted/VGG19_ImageNet_Caffe.model";
        }

        /// <summary>
        ///  Path to Cifar-10 model files
        /// </summary>
        public class Cifar10Path
        {
            public const string ResNet20 = "https://www.cntk.ai/Models/CNTK_Pretrained/ResNet20_CIFAR10_CNTK.model";
            public const string ResNet110 = "https://www.cntk.ai/Models/CNTK_Pretrained/ResNet110_CIFAR10_CNTK.model";
        }

        /// <summary>
        ///  Path to Fast RCNN model files
        /// </summary>
        public class FastRCNNPath
        {
            public const string Grocery100 = "https://www.cntk.ai/Models/FRCN_Grocery/Fast-RCNN_grocery100.model";
            public const string Pascal = "https://www.cntk.ai/Models/FRCN_Pascal/Fast-RCNN.model";
        }
       
    }

    public class DefaultPath
    {
        public static string Datasets = string.Format("{0}\\SiaNet\\dataset\\", Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData));
        public static string Models = string.Format("{0}\\SiaNet\\models\\", Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData));
    }
}
