using uint32_t = System.UInt32;
using uint64_t = System.UInt64;

// ReSharper disable once CheckNamespace
namespace SiaNet.Backend
{

    public sealed partial class Operators
    {

        #region Fields

        private static readonly string[] ConvolutionCudnnTuneValues =
        {
            "None",
            "fastest",
            "limited_workspace",
            "off"
        };

        private static readonly string[] ConvolutionLayoutValues =
        {
            "None",
            "NCDHW",
            "NCHW",
            "NCW",
            "NDHWC",
            "NHWC"
        };

        #endregion

        #region Methods

        public static Symbol Convolution(string symbolName,
                                         Symbol data,
                                         Symbol weight,
                                         Symbol bias,
                                         Shape kernel,
                                         uint32_t numFilter)
        {
            return Convolution(symbolName,
                               data,
                               weight,
                               bias,
                               kernel,
                               numFilter,
                               new Shape());

        }

        public static Symbol Convolution(string symbolName,
                                         Symbol data,
                                         Symbol weight,
                                         Symbol bias,
                                         Shape kernel,
                                         uint32_t numFilter,
                                         Shape stride)
        {
            return Convolution(symbolName,
                               data,
                               weight,
                               bias,
                               kernel,
                               numFilter,
                               stride,
                               new Shape());

        }

        public static Symbol Convolution(string symbolName,
                                         Symbol data,
                                         Symbol weight,
                                         Symbol bias,
                                         Shape kernel,
                                         uint32_t numFilter,
                                         Shape stride,
                                         Shape dilate)
        {
            return Convolution(symbolName,
                               data,
                               weight,
                               bias,
                               kernel,
                               numFilter,
                               stride,
                               dilate,
                               new Shape());

        }

        public static Symbol Convolution(string symbolName,
                                         Symbol data,
                                         Symbol weight,
                                         Symbol bias,
                                         Shape kernel,
                                         uint32_t numFilter,
                                         Shape stride,
                                         Shape dilate,
                                         Shape pad,
                                         uint32_t numGroup = 1,
                                         uint64_t workspace = 1024,
                                         bool noBias = false,
                                         ConvolutionCudnnTune cudnnTune = ConvolutionCudnnTune.None,
                                         bool cudnnOff = false,
                                         ConvolutionLayout layout = ConvolutionLayout.None)
        {
            return new Operator("Convolution").SetParam("kernel", kernel)
                                              .SetParam("num_filter", numFilter)
                                              .SetParam("stride", stride)
                                              .SetParam("dilate", dilate)
                                              .SetParam("pad", pad)
                                              .SetParam("num_group", numGroup)
                                              .SetParam("workspace", workspace)
                                              .SetParam("no_bias", noBias)
                                              .SetParam("cudnn_tune", ConvolutionCudnnTuneValues[(int)cudnnTune])
                                              .SetParam("cudnn_off", cudnnOff)
                                              .SetParam("layout", ConvolutionLayoutValues[(int)layout])
                                              .SetInput("data", data)
                                              .SetInput("weight", weight)
                                              .SetInput("bias", bias)
                                              .CreateSymbol(symbolName);
            
        }

        public static Symbol Convolution(Symbol data,
                                         Symbol weight,
                                         Symbol bias,
                                         Shape kernel,
                                         uint32_t numFilter)
        {
            return Convolution(data,
                               weight,
                               bias,
                               kernel,
                               numFilter,
                               new Shape());
        }

        public static Symbol Convolution(Symbol data,
                                         Symbol weight,
                                         Symbol bias,
                                         Shape kernel,
                                         uint32_t numFilter,
                                         Shape stride)
        {
            return Convolution(data,
                               weight,
                               bias,
                               kernel,
                               numFilter,
                               stride,
                               new Shape());
        }

        public static Symbol Convolution(Symbol data,
                                         Symbol weight,
                                         Symbol bias,
                                         Shape kernel,
                                         uint32_t numFilter,
                                         Shape stride,
                                         Shape dilate)
        {
            return Convolution(data,
                               weight,
                               bias,
                               kernel,
                               numFilter,
                               stride,
                               dilate,
                               new Shape());
        }
        
        public static Symbol Convolution(Symbol data,
                                         Symbol weight,
                                         Symbol bias,
                                         Shape kernel,
                                         uint32_t numFilter,
                                         Shape stride,
                                         Shape dilate,
                                         Shape pad,
                                         uint32_t numGroup = 1,
                                         uint64_t workspace = 1024,
                                         bool noBias = false,
                                         ConvolutionCudnnTune cudnnTune = ConvolutionCudnnTune.None,
                                         bool cudnnOff = false,
                                         ConvolutionLayout layout = ConvolutionLayout.None)
        {
            return new Operator("Convolution").SetParam("kernel", kernel)
                                              .SetParam("num_filter", numFilter)
                                              .SetParam("stride", stride)
                                              .SetParam("dilate", dilate)
                                              .SetParam("pad", pad)
                                              .SetParam("num_group", numGroup)
                                              .SetParam("workspace", workspace)
                                              .SetParam("no_bias", noBias)
                                              .SetParam("cudnn_tune", ConvolutionCudnnTuneValues[(int)cudnnTune])
                                              .SetParam("cudnn_off", cudnnOff)
                                              .SetParam("layout", ConvolutionLayoutValues[(int)layout])
                                              .SetInput("data", data)
                                              .SetInput("weight", weight)
                                              .SetInput("bias", bias)
                                              .CreateSymbol();
        }

        #endregion

    }

}
