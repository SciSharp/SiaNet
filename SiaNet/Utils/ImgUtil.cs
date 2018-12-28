using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;

namespace SiaNet
{
    public class ImgUtil
    {
        public static Tensor Im2Col(Tensor x, int h, int w, int padding=1, int stride = 1)
        {
            var x_padded = x.PadAll(1);

            return x;
        }
    }
}
