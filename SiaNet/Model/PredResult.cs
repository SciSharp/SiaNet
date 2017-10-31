using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet.Model
{
    public class PredResult
    {
        public PredResult()
        {
            DetectionBox = new BBox();
        }

        public string Name { get; set; }

        public double Score { get; set; }

        public BBox DetectionBox { get; set; }

    }

    public class BBox
    {
        public int X1 { get; set; }

        public int Y1 { get; set; }

        public int X2 { get; set; }

        public int Y2 { get; set; }
    }
}
