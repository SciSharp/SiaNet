using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet.Common
{
    /// <summary>
    /// Sample dataset list for tutorials
    /// </summary>
    public enum SampleDataset
    {
        /// <summary>
        /// Dataset taken from the StatLib library which is maintained at Carnegie Mellon University.\n Samples contain 13 attributes of houses at different locations around the Boston suburbs in the late 1970s. Targets are the median values of the houses at a location (in k$).
        /// </summary>
        HousingRegression,

        /// <summary>
        /// Dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images.
        /// </summary>
        MNIST,

        /// <summary>
        /// Dataset of 50,000 32x32 color training images, labeled over 10 categories, and 10,000 test images.
        /// </summary>
        Cifar10,

        /// <summary>
        /// Dataset of 50,000 32x32 color training images, labeled over 100 categories, and 10,000 test images.
        /// </summary>
        Cifar100,

        /// <summary>
        /// The Flowers dataset is a dataset for image classification created by the Visual Geometry Group at the University of Oxford. It consists of 102 different categories of flowers common to the UK
        /// </summary>
        Flowers,

        /// <summary>
        /// The Grocery dataset is a small toy data set that contains images of food items in a fridge.
        /// </summary>
        Grocery
    }

    public enum PrepDataset
    {
        /// <summary>
        /// The Pascal VOC dataset provides standardised image data sets for object class recognition.
        /// </summary>
        Pascal,

        /// <summary>
        /// ImageNet is an image database organized according to the WordNet hierarchy (currently only the nouns), in which each node of the hierarchy is depicted by hundreds and thousands of images.
        /// </summary>
        ImageNet
    }
}
