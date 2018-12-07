using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using SiaDNN;

namespace SiaNet.Backend
{
    public partial class NDArray
    {
        private static NDArray DoOperation(string name, NDArray data)
        {
            NDArray @out = new NDArray();
            new Operator(name)
                .SetInput("data", data)
                .Invoke(@out);
            return @out;
        }

        /// <summary>
        /// Computes the sum of array elements over given axes... Note::  `sum` and `sum_axis` are equivalent.Example::  data = [[[1,2],[2,3],[1,3]],          [[1,4],[4,3],[5,2]],          [[7,1],[7,2],[7,3]]]  sum(data, axis=1)  [[  4.   8.]   [ 10.   9.]   [ 21.   6.]]  sum(data, axis=[1,2])  [ 12.  19.  27.]Defined in G:\deeplearn\mxnet\src\operator\tensor\broadcast_reduce_op_value.cc:L69
        /// </summary>
        /// <param name="data">The input</param>
        /// <param name="axis">The axis or axes along which to perform the reduction.      The default, `axis=()`, will compute over all elements into a      scalar array with shape `(1,)`.      If `axis` is int, a reduction is performed on a particular axis.      If `axis` is a tuple of ints, a reduction is performed on all the axes      specified in the tuple.      If `exclude` is true, reduction will be performed on the axes that are      NOT in axis instead.      Negative values means indexing from right to left.</param>
        /// <param name="keepdims">If this is set to `True`, the reduced axes are left in the result as dimension with size one.</param>
        /// <param name="exclude">Whether to perform reduction on axis that are NOT in axis instead.</param>
        /// <returns>returns new symbol</returns>
        public static NDArray Sum(NDArray data, Shape axis = null, bool keepdims = false, bool exclude = false)
        {
            NDArray @out = new NDArray();
            if (axis == null)
                axis = new Shape();

            new Operator("sum")
            .SetParam("axis", axis)
            .SetParam("keepdims", keepdims)
            .SetParam("exclude", exclude)
            .SetInput("data", data)
            .Invoke(@out);
            return @out;
        }


        /// <summary>
        /// Returns element-wise squared value of the input... math::   square(x) = x^2Example::   square([2, 3, 4]) = [4, 9, 16]The storage type of ``square`` output depends upon the input storage type:   - square(default) = default   - square(row_sparse) = row_sparse   - square(csr) = csrDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L444
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <returns>returns new symbol</returns>
        public static NDArray Square(NDArray data)
        {
            return DoOperation("square", data);
        }

        /// <summary>
        /// Returns element-wise square-root value of the input... math::   \textrm{sqrt}(x) = \sqrt{x}Example::   sqrt([4, 9, 16]) = [2, 3, 4]The storage type of ``sqrt`` output depends upon the input storage type:   - sqrt(default) = default   - sqrt(row_sparse) = row_sparseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L467
        /// </summary>
        /// <param name="@out">output Ndarray</param>
        /// <param name="data">The input array.</param>
        /// <returns>returns new symbol</returns>
        public static NDArray Sqrt(NDArray data)
        {
            return DoOperation("sqrt", data);
        }

        /// <summary>
        /// Returns element-wise exponential value of the input... math::   exp(x) = e^x \approx 2.718^xExample::   exp([0, 1, 2]) = [1., 2.71828175, 7.38905621]The storage type of ``exp`` output is always denseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L543
        /// </summary>
        /// <param name="@out">output Ndarray</param>
        /// <param name="data">The input array.</param>
        /// <returns>returns new symbol</returns>
        public static NDArray Exp(NDArray data)
        {
            return DoOperation("exp", data);
        }
        
        /// <summary>
        /// Returns element-wise Natural logarithmic value of the input.The natural logarithm is logarithm in base *e*, so that ``log(exp(x)) = x``The storage type of ``log`` output is always denseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L555
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <returns>returns new symbol</returns>
        public static NDArray Log(NDArray data)
        {
            return DoOperation("log", data);
        }
        
        /// <summary>
        /// Returns element-wise Base-10 logarithmic value of the input.``10**log10(x) = x``The storage type of ``log10`` output is always denseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L567
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <returns>returns new symbol</returns>
        public static NDArray Log10(NDArray data)
        {
            return DoOperation("log10", data);
        }

        /// <summary>
        /// Returns element-wise Base-2 logarithmic value of the input.``2**log2(x) = x``The storage type of ``log2`` output is always denseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L579
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <returns>returns new symbol</returns>
        public static NDArray Log2(NDArray data)
        {
            return DoOperation("log2", data);
        }

        public static NDArray Clip(NDArray data, float a_min, float a_max)
        {
            NDArray @out = new NDArray();
            new Operator("clip")
            .SetParam("a_min", a_min)
            .SetParam("a_max", a_max)
            .SetInput("data", data)
            .Invoke(@out);

            return @out;
        }

        /// <summary>
        /// Casts all elements of the input to a new type... note:: ``Cast`` is deprecated. Use ``cast`` instead.Example::   cast([0.9, 1.3], dtype='int32') = [0, 1]   cast([1e20, 11.1], dtype='float16') = [inf, 11.09375]   cast([300, 11.1, 10.9, -1, -3], dtype='uint8') = [44, 11, 10, 255, 253]Defined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L218
        /// </summary>
        /// <param name="@out">output Ndarray</param>
        /// <param name="data">The input.</param>
        /// <param name="dtype">Output data type.</param>
        /// <returns>returns new symbol</returns>
        public static NDArray Cast(NDArray data, DType dtype)
        {
            NDArray @out = new NDArray();

            new Operator("cast")
            .SetParam("dtype", dtype)
            .SetInput("data", data)
            .Invoke(@out);

            return @out;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="lhs">first input</param>
        /// <param name="rhs">second input</param>
        /// <returns>returns new symbol</returns>
        public static NDArray GreaterEqual(NDArray lhs, NDArray rhs)
        {
            NDArray @out = new NDArray();

            new Operator("_greater_equal")
            .SetInput("lhs", lhs)
            .SetInput("rhs", rhs)
            .Invoke(@out);

            return @out;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="lhs">first input</param>
        /// <param name="rhs">second input</param>
        /// <returns>returns new symbol</returns>
        public static NDArray GreaterEqual(NDArray lhs, float rhs)
        {
            NDArray @out = new NDArray();
            float[] rhsdata = new float[lhs.Size];
            for (ulong i = 0; i < lhs.Size; i++)
            {
                rhsdata[i] = rhs;
            }


            NDArray rhsArray = new NDArray(rhsdata, new Shape(lhs.GetShape()));

            new Operator("_greater_equal")
            .SetInput("lhs", lhs)
            .SetInput("rhs", rhsArray)
            .Invoke(@out);

            return @out;
        }

        public static NDArray OneHot(NDArray data, int depth, double on_value = 1, double off_value = 0, DType dtype = null)
        {
            NDArray @out = new NDArray();
            if (dtype == null) { dtype = DType.Float32; }

            new Operator("one_hot")
            .SetParam("depth", depth)
            .SetParam("on_value", on_value)
            .SetParam("off_value", off_value)
            .SetParam("dtype", dtype)
            .SetInput("indices", data)
            .Invoke(@out);

            return @out;
        }


    }
}
