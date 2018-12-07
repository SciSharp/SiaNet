using mx_float = System.Single;

// ReSharper disable once CheckNamespace
namespace SiaNet.Backend
{

    public sealed partial class Operators
    {

        #region Fields

        private static readonly string[] PoolingPoolTypeValues =
        {
            "avg",
            "max",
            "sum"
        };

        private static readonly string[] PoolingPoolingConventionValues =
        {
            "full",
            "valid"
        };

        #endregion

        #region Methods

        public static Symbol Pooling(string symbolName,
                                     Symbol data,
                                     Shape kernel,
                                     PoolingPoolType poolType,
                                     bool globalPool = false,
                                     bool cudnnOff = false,
                                     PoolingPoolingConvention poolingConvention = PoolingPoolingConvention.Valid)
        {
            return Pooling(symbolName, data, kernel, poolType, globalPool, cudnnOff, poolingConvention, new Shape());
        }

        public static Symbol Pooling(string symbolName,
                                     Symbol data,
                                     Shape kernel,
                                     PoolingPoolType poolType,
                                     bool globalPool,
                                     bool cudnnOff,
                                     PoolingPoolingConvention poolingConvention,
                                     Shape stride)
        {
            return Pooling(symbolName, data, kernel, poolType, globalPool, cudnnOff, poolingConvention, stride, new Shape());
        }

        public static Symbol Pooling(string symbolName,
                                     Symbol data,
                                     Shape kernel,
                                     PoolingPoolType poolType,
                                     bool globalPool,
                                     bool cudnnOff,
                                     PoolingPoolingConvention poolingConvention,
                                     Shape stride,
                                     Shape pad)
        {
            return new Operator("Pooling").SetParam("kernel", kernel)
                                          .SetParam("pool_type", PoolingPoolTypeValues[(int)poolType])
                                          .SetParam("global_pool", globalPool)
                                          .SetParam("cudnn_off", cudnnOff)
                                          .SetParam("pooling_convention", PoolingPoolingConventionValues[(int)poolingConvention])
                                          .SetParam("stride", stride)
                                          .SetParam("pad", pad)
                                          .SetInput("data", data)
                                          .CreateSymbol(symbolName);
        }

        public static Symbol Pooling(Symbol data,
                                     Shape kernel,
                                     PoolingPoolType poolType,
                                     bool globalPool = false,
                                     bool cudnnOff = false,
                                     PoolingPoolingConvention poolingConvention = PoolingPoolingConvention.Valid)
        {
            return Pooling(data, kernel, poolType, globalPool, cudnnOff, poolingConvention, new Shape());
        }

        public static Symbol Pooling(Symbol data,
                                     Shape kernel,
                                     PoolingPoolType poolType,
                                     bool globalPool,
                                     bool cudnnOff,
                                     PoolingPoolingConvention poolingConvention,
                                     Shape stride)
        {
            return Pooling(data, kernel, poolType, globalPool, cudnnOff, poolingConvention, stride, new Shape());
        }

        public static Symbol Pooling(Symbol data,
                                     Shape kernel,
                                     PoolingPoolType poolType,
                                     bool globalPool,
                                     bool cudnnOff,
                                     PoolingPoolingConvention poolingConvention,
                                     Shape stride,
                                     Shape pad)
        {
            return new Operator("Pooling").SetParam("kernel", kernel)
                                          .SetParam("pool_type", PoolingPoolTypeValues[(int)poolType])
                                          .SetParam("global_pool", globalPool)
                                          .SetParam("cudnn_off", cudnnOff)
                                          .SetParam("pooling_convention", PoolingPoolingConventionValues[(int)poolingConvention])
                                          .SetParam("stride", stride)
                                          .SetParam("pad", pad)
                                          .SetInput("data", data)
                                          .CreateSymbol();
        }

        #endregion

    }

}
