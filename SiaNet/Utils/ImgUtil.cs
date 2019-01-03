using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;

namespace SiaNet
{
    public class ImgUtil
    {
        public static Tensor Im2Col(Tensor x, uint field_height, uint field_width, uint field_depth, uint? padding = null, uint stride = 1)
        {
            var (i, j, k) = Im2ColIndices(x.Shape, field_height, field_width, padding, stride);

            if (padding.HasValue)
                x.PadAll(padding.Value);


            //  p = padding
            //x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode = 'constant')

            //k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
            //                             stride)

            //cols = x_padded[:, k, i, j]
            //C = x.shape[1]
            //cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
            //return cols

            return x;
        }

        public static Tensor Im2Col(Tensor x, uint field_height, uint field_width, uint? padding=null, uint stride = 1)
        {
            var (i, j, k) = Im2ColIndices(x.Shape, field_height, field_width, padding, stride);

            if(padding.HasValue)
                x.PadAll(padding.Value);


            //  p = padding
            //x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode = 'constant')

            //k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
            //                             stride)

            //cols = x_padded[:, k, i, j]
            //C = x.shape[1]
            //cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
            //return cols

            return x;
        }

        public static Tensor Im2Col(Tensor x, uint field_height, uint? padding = null, uint stride = 1)
        {

            //  p = padding
            //x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode = 'constant')

            //k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
            //                             stride)

            //cols = x_padded[:, k, i, j]
            //C = x.shape[1]
            //cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
            //return cols

            return x;
        }

        public static Tensor Col2Im(Tensor cols, long[] x_shape, uint field_height, uint field_width, uint field_depth, uint? padding = null, uint stride = 1)
        {
            return cols;
        }

        public static Tensor Col2Im(Tensor cols, long[] x_shape, uint field_height, uint field_width, uint? padding = null, uint stride = 1)
        {
            return cols;
        }

        public static Tensor Col2Im(Tensor cols, long[] x_shape, uint field_height, uint? padding = null, uint stride = 1)
        {
            return cols;
        }

        private static ValueTuple<Tensor, Tensor, Tensor> Im2ColIndices(long[] shape, uint field_height = 3, uint field_width=3, uint? padding=null, uint stride=1)
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

            var i0 = TOps.Repeat(Tensor.Arange(Global.Device, 0, field_height), (int)field_width);
            i0 = TOps.Tile(i0, C);
            var i1 = stride * TOps.Repeat(Tensor.Arange(Global.Device, 0, (int)out_height), (int)out_width);
           
            var j0 = TOps.Tile(Tensor.Arange(Global.Device, 0, field_width), field_height * C);
            var j1 = stride * TOps.Tile(Tensor.Arange(Global.Device, 0, (int)out_width), (int)out_height);
            
            var i = i0.Reshape(-1, 1) + i1.Reshape(1, -1);
            var j = j0.Reshape(-1, 1) + j1.Reshape(1, -1);

            var k = TOps.Repeat(Tensor.Arange(Global.Device, 0, C), (int)field_height * (int)field_width);
            k = k.View(k.ElementCount(), 1);
            return (i, j, k);
        }
    }
}
