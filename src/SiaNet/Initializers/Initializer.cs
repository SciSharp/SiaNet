using SiaNet.Backend;
using System;

// ReSharper disable once CheckNamespace
namespace SiaDNN.Initializers
{

    public abstract class BaseInitializer
    {

        #region Methods

        public virtual void Operator(string name, NDArray array)
        {
            if (StringStartWith(name, "upsampling"))
            {
                InitBilinear(array);
            }
            else if (StringEndWith(name, "bias"))
            {
                InitBias(array);
            }
            else if (StringEndWith(name, "gamma"))
            {
                InitGamma(array);
            }
            else if (StringEndWith(name, "beta"))
            {
                InitBeta(array);
            }
            else if (StringEndWith(name, "weight"))
            {
                InitWeight(array);
            }
            else if (StringEndWith(name, "moving_mean"))
            {
                InitZero(array);
            }
            else if (StringEndWith(name, "moving_var"))
            {
                InitOne(array);
            }
            else if (StringEndWith(name, "moving_inv_var"))
            {
                InitZero(array);
            }
            else if (StringEndWith(name, "moving_avg"))
            {
                InitZero(array);
            }
            else
            {
                InitDefault(array);
            }
        }

        public static bool StringStartWith(string name, string check)
        {
            return name.Length >= check.Length && name.Substring(0, check.Length) == check;
        }

        public static bool StringEndWith(string name, string check)
        {
            return name.Length >= check.Length &&
                   name.Substring(name.Length - check.Length, check.Length) == check;
        }

        #region Overrids

        protected virtual void InitBeta(NDArray array)
        {
            array.Set(0.0f);
        }

        protected virtual void InitBias(NDArray array)
        {
            array.Set(0.0f);
        }

        protected virtual void InitBilinear(NDArray array)
        {
            var shape = new Shape(array.GetShape());
            var size = (int)shape.Size;
            var weight = new float[size];
            var f = Math.Ceiling(shape[3] / 2.0f);
            var c = (2 * f - 1 - f % 2) / (2.0 * f);

            for (var i = 0; i < size; ++i)
            {
                var x = i % shape[3];
                var y = (i / shape[3]) % shape[2];
                weight[i] = (float)((1 - Math.Abs(x / f - c)) * (1 - Math.Abs(y / f - c)));
            }

            array.SyncCopyFromCPU(weight);
        }

        protected virtual void InitDefault(NDArray array)
        {

        }

        protected virtual void InitGamma(NDArray array)
        {
            array.Set(1.0f);
        }

        protected virtual void InitOne(NDArray array)
        {
            array.Set(1.0f);
        }

        protected virtual void InitWeight(NDArray array)
        {

        }

        protected virtual void InitZero(NDArray array)
        {
            array.Set(1.0f);
        }

        #endregion

        #endregion

    }

}
