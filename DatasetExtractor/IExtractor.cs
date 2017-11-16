using SiaNet.Model;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DatasetExtractor
{
    internal interface IExtractor
    {
        XYFrame TrainFrame { get; set; }

        XYFrame TestFrame { get; set; }

        void Download();

        void Extract();
    }
}
