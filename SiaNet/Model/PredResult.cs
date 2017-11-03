using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet.Model
{
    public class PredResult
    {
        public string Name { get; set; }

        public double Score { get; set; }

        public Rectangle BBox { get; set; }

    }

    public class BBox
    {
        public int X { get; set; }

        public int Y { get; set; }

        public int W { get; set; }

        public int H { get; set; }
    }
}
