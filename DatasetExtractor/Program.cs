using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DatasetExtractor
{
    class Program
    {
        static void Main(string[] args)
        {
            IExtractor extractor = new MNISTExtractor();
            extractor.Download();
            extractor.Extract();

            extractor = new Cifar10Extractor();
            extractor.Download();
            extractor.Extract();

            extractor = new Cifar100Extractor();
            extractor.Download();
            extractor.Extract();
        }
    }
}
