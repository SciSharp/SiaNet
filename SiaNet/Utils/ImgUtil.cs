using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;

namespace SiaNet
{
    public class ImgUtil
    {
        public static Tensor Im2Col(Tensor x, int field_height, int field_width, int padding=1, int stride = 1)
        {
            Tensor x_padded = x.View(x.Shape);

            x_padded = x_padded.PadAll(1);
            x_padded.Print("padded");
            //  p = padding
            //x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode = 'constant')

            //k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
            //                             stride)

            //cols = x_padded[:, k, i, j]
            //C = x.shape[1]
            //cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
            //return cols

            var (i, j, k) = Im2ColIndices(x.Shape, field_height, field_width, padding, stride);
            i.Print("i");
            j.Print("j");
            k.Print("k");
            return x;
        }

        private static ValueTuple<Tensor, Tensor, Tensor> Im2ColIndices(long[] shape, int field_height = 3, int field_width=3, int padding=1, int stride=1)
        {
            long N = shape[0];
            long C = shape[1];
            long H = shape[2];
            long W = shape[3];

            if((H + 2 * padding - field_height) % stride != 0)
            {
                throw new ArgumentException("Invalid field_height");
            }

            if ((W + 2 * padding - field_width) % stride != 0)
            {
                throw new ArgumentException("Invalid field_width");
            }

            var out_height = (H + 2 * padding - field_height) / stride + 1;
            var out_width = (W + 2 * padding - field_width) / stride + 1;

            var i0 = TOps.Repeat(Tensor.Arange(Global.Device, 1, field_height), field_width);
            i0 = TOps.Tile(i0, C);
            var i1 = stride * TOps.Repeat(Tensor.Arange(Global.Device, 1, out_height), (int)out_width);
           
            var j0 = TOps.Tile(Tensor.Arange(Global.Device, 1, field_width), field_height * C);
            var j1 = stride * TOps.Tile(Tensor.Arange(Global.Device, 1, out_width), (int)out_height);
            
            var i = i0.Reshape(-1, 1) + i1.Reshape(1, -1);
            var j = j0.Reshape(-1, 1) + j1.Reshape(1, -1);

            var k = TOps.Repeat(Tensor.Arange(Global.Device, 1, C), field_height * field_width);
            k = k.View(k.ElementCount(), 1);

            i0.Print();
            i1.Print();
            j0.Print(); j1.Print();
            return (i, j, k);
        }
    }
}
