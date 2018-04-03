using System;

namespace SiaNet.Common
{
    /// <summary>
    ///     Image Net pretrained model list
    /// </summary>
    public enum ImageNetModel
    {
        /// <summary>
        ///     The Alexnet model
        /// </summary>
        AlexNet,

        /// <summary>
        ///     The inception v3 model
        /// </summary>
        InceptionV3,

        /// <summary>
        ///     The ResNet18 model
        /// </summary>
        ResNet18,

        /// <summary>
        ///     The ResNet34 model
        /// </summary>
        ResNet34,

        /// <summary>
        ///     The ResNet50 model
        /// </summary>
        ResNet50,

        /// <summary>
        ///     The ResNet101 model
        /// </summary>
        ResNet101,

        /// <summary>
        ///     The ResNet152 model
        /// </summary>
        ResNet152,

        /// <summary>
        ///     The VGG16 model
        /// </summary>
        VGG16,

        /// <summary>
        ///     The VGG19
        /// </summary>
        VGG19
    }

    /// <summary>
    ///     Cifar-10 pretrained model list
    /// </summary>
    public enum Cifar10Model
    {
        /// <summary>
        ///     Cifar-10 ResNet20 model
        /// </summary>
        ResNet20,

        /// <summary>
        ///     Cifar-10 ResNet110 model
        /// </summary>
        ResNet110
    }

    /// <summary>
    ///     Fast-RCNN pretrained model list
    /// </summary>
    public enum FastRCNNModel
    {
        /// <summary>
        ///     Fast-RCNN Grocery100 model
        /// </summary>
        Grocery100,

        /// <summary>
        ///     Fast-RCNN Pascal model
        /// </summary>
        Pascal
    }

    /// <summary>
    ///     Pre trained model path to download from hosted sites
    /// </summary>
    public class PreTrainedModelPath
    {
        /// <summary>
        ///     Path to Cifar-10 model files
        /// </summary>
        public class Cifar10Path
        {
            public const string ResNet110 = "https://www.cntk.ai/Models/CNTK_Pretrained/ResNet110_CIFAR10_CNTK.model";
            public const string ResNet20 = "https://www.cntk.ai/Models/CNTK_Pretrained/ResNet20_CIFAR10_CNTK.model";
        }

        /// <summary>
        ///     Path to Fast RCNN model files
        /// </summary>
        public class FastRCNNPath
        {
            public const string Grocery100 = "https://www.cntk.ai/Models/FRCN_Grocery/Fast-RCNN_grocery100.model";
            public const string Pascal = "https://www.cntk.ai/Models/FRCN_Pascal/Fast-RCNN.model";
        }

        /// <summary>
        ///     Path to ImageNet model files for various models like AlexNet, ResNet, VGG etc..
        /// </summary>
        public class ImageNetPath
        {
            public const string AlexNet = "https://www.cntk.ai/Models/CNTK_Pretrained/AlexNet_ImageNet_CNTK.model";

            public const string InceptionV3 =
                "https://www.cntk.ai/Models/CNTK_Pretrained/InceptionV3_ImageNet_CNTK.model";

            public const string ResNet101 = "https://www.cntk.ai/Models/Caffe_Converted/ResNet101_ImageNet_Caffe.model";
            public const string ResNet152 = "https://www.cntk.ai/Models/Caffe_Converted/ResNet152_ImageNet_Caffe.model";
            public const string ResNet18 = "https://www.cntk.ai/Models/CNTK_Pretrained/ResNet18_ImageNet_CNTK.model";
            public const string ResNet34 = "https://www.cntk.ai/Models/CNTK_Pretrained/ResNet34_ImageNet_CNTK.model";
            public const string ResNet50 = "https://www.cntk.ai/Models/CNTK_Pretrained/ResNet50_ImageNet_CNTK.model";
            public const string VGG16 = "https://www.cntk.ai/Models/Caffe_Converted/VGG16_ImageNet_Caffe.model";
            public const string VGG19 = "https://www.cntk.ai/Models/Caffe_Converted/VGG19_ImageNet_Caffe.model";
        }
    }

    public class DefaultPath
    {
        public static string Datasets = string.Format("{0}\\SiaNet\\dataset\\",
            Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData));

        public static string Models = string.Format("{0}\\SiaNet\\models\\",
            Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData));
    }
}