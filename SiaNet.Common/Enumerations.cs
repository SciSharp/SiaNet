using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet.Common
{
    public enum SampleDataset
    {
        HousingRegression,
        MNIST,
        Cifar10,
        Cifar100,
        Flowers,
        Grocery
    }

    public enum PrepDataset
    {
        Pascal,
        ImageNet
    }
}
