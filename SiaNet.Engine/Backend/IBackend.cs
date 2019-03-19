using SiaNet.Engine;
using SiaNet.Engine.Layers;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Text;

namespace SiaNet.Engine
{
    public interface IBackend
    {
        /// <summary>
        /// Epsilon which is the smallest float value and > 0
        /// </summary>
        /// <returns></returns>
        float Epsilon();

        /// <summary>
        /// Gets the shape of the tensor.
        /// </summary>
        /// <param name="x">The x.</param>
        /// <returns></returns>
        long[] GetShape(Tensor x);

        /// <summary>
        /// Evals the specified tensor.
        /// </summary>
        /// <param name="x">The x.</param>
        /// <returns></returns>
        object Eval(Tensor x);

        /// <summary>
        /// Prints the specified tensor for observation.
        /// </summary>
        /// <param name="x">The x.</param>
        /// <param name="title">The title.</param>
        void Print(Tensor x, string title = "");

        /// <summary>
        /// Generate the unique identifier for layers and parameters
        /// </summary>
        /// <param name="name">The name.</param>
        /// <returns></returns>
        string UUID(string name);

        /// <summary>
        /// Sets the device CPU, GPU or OpenCL.
        /// </summary>
        /// <param name="device">The device.</param>
        void SetDevice(DeviceType device);

        /// <summary>
        /// Gets the type of the data.
        /// </summary>
        /// <param name="x">The x.</param>
        /// <returns></returns>
        DataType GetDataType(Tensor x);

        #region Initializers
        /// <summary>
        /// Creates the variable.
        /// </summary>
        /// <param name="data">The data.</param>
        /// <param name="shape">The shape.</param>
        /// <param name="name">The name.</param>
        /// <returns></returns>
        Tensor CreateVariable(float[] data, long[] shape, string name = "");

        /// <summary>
        /// Reshapes the specified x.
        /// </summary>
        /// <param name="x">The x.</param>
        /// <param name="shape">The shape.</param>
        /// <returns></returns>
        Tensor Reshape(Tensor x, params long[] shape);

        /// <summary>
        /// Create a constant variable with specified value and target tensor shape
        /// </summary>
        /// <param name="value">The value.</param>
        /// <param name="shape">The shape.</param>
        /// <returns></returns>
        Tensor Constant(float value, long[] shape);

        /// <summary>
        /// Generate variable with Randon Uniform distribution data
        /// </summary>
        /// <param name="shape">The shape.</param>
        /// <param name="min">The minimum.</param>
        /// <param name="max">The maximum.</param>
        /// <param name="seed">The seed.</param>
        /// <returns></returns>
        Tensor RandomUniform(long[] shape, float min, float max, int? seed = null);

        /// <summary>
        /// Generate variable with Randon Normal distribution data
        /// </summary>
        /// <param name="shape">The shape.</param>
        /// <param name="mean">The mean.</param>
        /// <param name="stddev">The stddev.</param>
        /// <param name="seed">The seed.</param>
        /// <returns></returns>
        Tensor RandomNormal(long[] shape, float mean, float stddev, int? seed = null);

        /// <summary>
        /// Generate variable with Randon Bernoulli distribution 
        /// </summary>
        /// <param name="shape">The shape.</param>
        /// <param name="p">The probability value</param>
        /// <returns></returns>
        Tensor RandomBernoulli(long[] shape, float p);
        #endregion

        #region BasicOps
        /// <summary>
        /// Adds two tensor
        /// </summary>
        /// <param name="a">Tensor A</param>
        /// <param name="b">Tensor B</param>
        /// <returns></returns>
        Tensor Add(Tensor a, Tensor b);

        /// <summary>
        /// Adds Tensor and float value
        /// </summary>
        /// <param name="a">Tensor data</param>
        /// <param name="b">Float Value</param>
        /// <returns></returns>
        Tensor Add(Tensor a, float b);

        /// <summary>
        /// Adds float and Tensor
        /// </summary>
        /// <param name="a">Float value</param>
        /// <param name="b">Tensor data</param>
        /// <returns></returns>
        Tensor Add(float a, Tensor b);

        /// <summary>
        /// Subtract two tensor
        /// </summary>
        /// <param name="a">First Tensor</param>
        /// <param name="b">Second Tensor</param>
        /// <returns></returns>
        Tensor Sub(Tensor a, Tensor b);

        /// <summary>
        /// Subtract tensor and float
        /// </summary>
        /// <param name="a">Tensor data</param>
        /// <param name="b">Float value</param>
        /// <returns></returns>
        Tensor Sub(Tensor a, float b);

        /// <summary>
        /// Subtract float and tensor
        /// </summary>
        /// <param name="a">Float value</param>
        /// <param name="b">Tensor data</param>
        /// <returns></returns>
        Tensor Sub(float a, Tensor b);

        /// <summary>
        /// Multiply two tensor
        /// </summary>
        /// <param name="a">First tensor</param>
        /// <param name="b">Second tensor</param>
        /// <returns></returns>
        Tensor Mul(Tensor a, Tensor b);

        /// <summary>
        /// Multiply tensor and float
        /// </summary>
        /// <param name="a">Tensor data.</param>
        /// <param name="b">Float value</param>
        /// <returns></returns>
        Tensor Mul(Tensor a, float b);

        /// <summary>
        /// Multiply float and tensor
        /// </summary>
        /// <param name="a">Float value.</param>
        /// <param name="b">Tensor data.</param>
        /// <returns></returns>
        Tensor Mul(float a, Tensor b);

        /// <summary>
        /// Divide two tensor
        /// </summary>
        /// <param name="a">First tensor.</param>
        /// <param name="b">Second tensor.</param>
        /// <returns></returns>
        Tensor Div(Tensor a, Tensor b);

        /// <summary>
        /// Divide tensor with float
        /// </summary>
        /// <param name="a">Tensor data.</param>
        /// <param name="b">Float value.</param>
        /// <returns></returns>
        Tensor Div(Tensor a, float b);

        /// <summary>
        /// Divide float with tensor
        /// </summary>
        /// <param name="a">Float value.</param>
        /// <param name="b">Tensor data.</param>
        /// <returns></returns>
        Tensor Div(float a, Tensor b);

        /// <summary>
        /// Perform a > b between two tensor
        /// </summary>
        /// <param name="a">First tensor</param>
        /// <param name="b">Second tensor.</param>
        /// <returns></returns>
        Tensor GreaterThan(Tensor a, Tensor b);

        /// <summary>
        /// Perform a >= b between two tensor
        /// </summary>
        /// <param name="a">First tensor</param>
        /// <param name="b">Second tensor.</param>
        /// <returns></returns>
        Tensor GreaterThanEqual(Tensor a, Tensor b);

        /// <summary>
        /// <![CDATA[Perform a < b between two tensor]]>
        /// </summary>
        /// <param name="a">First tensor</param>
        /// <param name="b">Second tensor.</param>
        /// <returns></returns>
        Tensor LessThan(Tensor a, Tensor b);

        /// <summary>
        /// <![CDATA[Perform a <= b between two tensor]]>
        /// </summary>
        /// <param name="a">First tensor</param>
        /// <param name="b">Second tensor.</param>
        /// <returns></returns>
        Tensor LessThanEqual(Tensor a, Tensor b);

        /// <summary>
        /// Perform a > b between float and tensor
        /// </summary>
        /// <param name="a">Float value</param>
        /// <param name="b">Tensor data.</param>
        /// <returns></returns>
        Tensor GreaterThan(float a, Tensor b);

        /// <summary>
        /// Perform a >= b between float and tensor
        /// </summary>
        /// <param name="a">Float value</param>
        /// <param name="b">Tensor data.</param>
        /// <returns></returns>
        Tensor GreaterThanEqual(float a, Tensor b);

        /// <summary>
        /// <![CDATA[Perform a < b between float and Tensor]]>
        /// </summary>
        /// <param name="a">Floaty value</param>
        /// <param name="b">Tensor data.</param>
        /// <returns></returns>
        Tensor LessThan(float a, Tensor b);

        /// <summary>
        /// <![CDATA[Perform a <= b between float and Tensor]]>
        /// </summary>
        /// <param name="a">Tensor data</param>
        /// <param name="b">Float value.</param>
        /// <returns></returns>
        Tensor LessThanEqual(float a, Tensor b);

        /// <summary>
        /// Perform a > b between tensor and float
        /// </summary>
        /// <param name="a">Tensor data</param>
        /// <param name="b">Float value.</param>
        /// <returns></returns>
        Tensor GreaterThan(Tensor a, float b);

        /// <summary>
        /// Perform a >= b between tensor and float
        /// </summary>
        /// <param name="a">Tensor data</param>
        /// <param name="b">Float value.</param>
        /// <returns></returns>
        Tensor GreaterThanEqual(Tensor a, float b);

        /// <summary>
        /// <![CDATA[Perform a < b between tensor and float]]>
        /// </summary>
        /// <param name="a">Tensor data</param>
        /// <param name="b">Float value.</param>
        /// <returns></returns>
        Tensor LessThan(Tensor a, float b);

        /// <summary>
        /// <![CDATA[Perform a <= b between tensor and float]]>
        /// </summary>
        /// <param name="a">Tensor data</param>
        /// <param name="b">Float value.</param>
        /// <returns></returns>
        Tensor LessThanEqual(Tensor a, float b);

        /// <summary>
        /// Check if two tensor are equal
        /// </summary>
        /// <param name="a">First tensor.</param>
        /// <param name="b">Second Tensor.</param>
        /// <returns></returns>
        Tensor EqualTo(Tensor a, Tensor b);

        /// <summary>
        /// Construct an array by repeating A the number of times given by reps.
        /// </summary>
        /// <param name="x">Tensor to be repeated.</param>
        /// <param name="n">Number of time.</param>
        /// <param name="axis">The axis to be repeated.</param>
        /// <returns></returns>
        Tensor Tile(Tensor x, int n, int axis = 0);
        #endregion

        #region Math Ops
        /// <summary>
        /// Return the absolute value element-wise.
        /// </summary>
        /// <param name="x">The tensor.</param>
        /// <returns></returns>
        Tensor Abs(Tensor x);

        /// <summary>
        /// Negate the value element-wise
        /// </summary>
        /// <param name="x">The tensor.</param>
        /// <returns></returns>
        Tensor Neg(Tensor x);

        /// <summary>
        /// Return the square root value element-wise.
        /// </summary>
        /// <param name="x">The tensor.</param>
        /// <returns></returns>
        Tensor Sqrt(Tensor x);

        /// <summary>
        /// Return the exponential value element-wise.
        /// </summary>
        /// <param name="x">The tensor.</param>
        /// <returns></returns>
        Tensor Exp(Tensor x);

        /// <summary>
        /// Calculate the logrithmic value element-wise.
        /// </summary>
        /// <param name="x">The tensor.</param>
        /// <returns></returns>
        Tensor Log(Tensor x);

        /// <summary>
        /// Return the base 10 logarithm of the input array, element-wise.
        /// </summary>
        /// <param name="x">The tensor.</param>
        /// <returns></returns>
        Tensor Log10(Tensor x);

        /// <summary>
        /// Return the natural logarithm of one plus the input array, element-wise.
        /// </summary>
        /// <param name="x">The tensor.</param>
        /// <returns></returns>
        Tensor Log1p(Tensor x);

        /// <summary>
        /// Calculate the floor value element-wise.
        /// </summary>
        /// <param name="x">The tensor.</param>
        /// <returns></returns>
        Tensor Floor(Tensor x);

        /// <summary>
        /// Calculate the ceil value element-wise.
        /// </summary>
        /// <param name="x">The tensor.</param>
        /// <returns></returns>
        Tensor Ceil(Tensor x);

        /// <summary>
        /// Calculate the rounded to nearest integer value element-wise.
        /// </summary>
        /// <param name="x">The tensor.</param>
        /// <returns></returns>
        Tensor Round(Tensor x);

        /// <summary>
        /// Calculate the sin value element-wise.
        /// </summary>
        /// <param name="x">The tensor.</param>
        /// <returns></returns>
        Tensor Sin(Tensor x);

        /// <summary>
        /// Calculate the cos value element-wise.
        /// </summary>
        /// <param name="x">The tensor.</param>
        /// <returns></returns>
        Tensor Cos(Tensor x);

        /// <summary>
        /// Calculate the tan value element-wise.
        /// </summary>
        /// <param name="x">The tensor.</param>
        /// <returns></returns>
        Tensor Tan(Tensor x);

        /// <summary>
        /// Calculate the invert sign value element-wise.
        /// </summary>
        /// <param name="x">The tensor.</param>
        /// <returns></returns>
        Tensor Asin(Tensor x);

        /// <summary>
        /// Calculate the invert cos value element-wise.
        /// </summary>
        /// <param name="x">The tensor.</param>
        /// <returns></returns>
        Tensor Acos(Tensor x);

        /// <summary>
        /// Calculate the invert tan value element-wise.
        /// </summary>
        /// <param name="x">The tensor.</param>
        /// <returns></returns>
        Tensor Atan(Tensor x);

        /// <summary>
        /// Calculate the hyperbolic sin value element-wise.
        /// </summary>
        /// <param name="x">The tensor.</param>
        /// <returns></returns>
        Tensor Sinh(Tensor x);

        /// <summary>
        /// Calculate the hyperbolic cos value element-wise.
        /// </summary>
        /// <param name="x">The tensor.</param>
        /// <returns></returns>
        Tensor Cosh(Tensor x);

        /// <summary>
        /// Calculate the hyperbolic tan value element-wise.
        /// </summary>
        /// <param name="x">The tensor.</param>
        /// <returns></returns>
        Tensor Tanh(Tensor x);

        /// <summary>
        /// Calculate the sigmoid value element-wise.
        /// </summary>
        /// <param name="x">The tensor.</param>
        /// <returns></returns>
        Tensor Sigmoid(Tensor x);

        /// <summary>
        /// Power up the specified tensor.
        /// </summary>
        /// <param name="x">The tensor.</param>
        /// <param name="value">The power up value.</param>
        /// <returns></returns>
        Tensor Pow(Tensor x, float value);

        /// <summary>
        /// Calculate the square of tensor element-wise
        /// </summary>
        /// <param name="x">The tensor.</param>
        /// <returns></returns>
        Tensor Square(Tensor x);

        /// <summary>
        /// Clips the specified tensor between min and max value, elemen-wise.
        /// </summary>
        /// <param name="x">The tensor data.</param>
        /// <param name="min">The minimum value.</param>
        /// <param name="max">The maximum value.</param>
        /// <returns></returns>
        Tensor Clip(Tensor x, float min, float max);
        #endregion

        #region Aggregate Ops
        /// <summary>
        /// Sum of array elements overall axis
        /// </summary>
        /// <param name="x">The tensor</param>
        /// <returns></returns>
        float Sum(Tensor x);

        /// <summary>
        /// Sum of array elements over a given axis.
        /// </summary>
        /// <param name="x">The tensor.</param>
        /// <param name="dim">The dim to reduce.</param>
        /// <returns></returns>
        Tensor Sum(Tensor x, int dim);

        /// <summary>
        /// Sum of array elements over multiple axis.
        /// </summary>
        /// <param name="x">The tensor.</param>
        /// <param name="dim">The dimensions list.</param>
        /// <returns></returns>
        Tensor Sum(Tensor x, params int[] dim);

        float Max(Tensor x);

        Tensor Max(Tensor x, int dim);

        Tensor Max(Tensor x, params int[] dim);

        float Min(Tensor x);

        Tensor Min(Tensor x, int dim);

        Tensor Min(Tensor x, params int[] dim);

        float Mean(Tensor x);

        Tensor Mean(Tensor x, int dim);

        Tensor Mean(Tensor x, params int[] dim);

        Tensor Argmax(Tensor x, int dim = 0);

        Tensor Argmin(Tensor x, int dim = 0);

        Tensor Maximum(Tensor a, Tensor b);

        Tensor Maximum(Tensor a, float b);

        Tensor Minimum(Tensor a, Tensor b);

        Tensor Minimum(Tensor a, float b);
        #endregion

        #region Matrix Ops
        Tensor Transpose(Tensor x);

        Tensor Transpose(Tensor x, params int[] dims);

        Tensor Dot(Tensor a, Tensor b);

        Tensor Diag(Tensor x);
        #endregion

        #region NN Ops
        Tensor Softmax(Tensor x, int axis = -1);

        Tensor Softplus(Tensor x, int axis = -1);

        Tensor L2Normalize(Tensor x, int axis = -1);

        Tensor Im2Col(Tensor x, Tuple<int, int> kernalSize, int padding = 1, int stride = 1);

        Tensor Col2Im(Tensor cols, long[] x_shape, Tuple<int, int> kernalSize, int padding = 1, int stride = 1);
        #endregion

        #region Misc
        Tensor SliceRows(Tensor x, long start, long end);

        Tensor SliceCols(Tensor x, long start, long end);

        Array GetArray(Tensor x);

        void Dispose(Tensor x);

        ActivationFunc GetActFunc();
        #endregion
    }
}
