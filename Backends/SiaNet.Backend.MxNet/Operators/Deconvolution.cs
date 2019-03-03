using uint32_t = System.UInt32;
using uint64_t = System.UInt64;

// ReSharper disable once CheckNamespace
namespace SiaNet.Backend.MxNetLib
{

    public sealed partial class Operators
    {

        #region Fields

        private static readonly string[] DeconvolutionCudnnTuneValues =
        {
            "None",
            "fastest",
            "limited_workspace",
            "off"
        };

        private static readonly string[] DeconvolutionLayoutValues =
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

        public static Symbol Deconvolution(string symbolName,
                                           Symbol data,
                                           Symbol weight,
                                           Symbol bias,
                                           Shape kernel,
                                           uint32_t numFilter)
        {
            return Deconvolution(symbolName,
                                 data,
                                 weight,
                                 bias,
                                 kernel,
                                 numFilter,
                                 new Shape());
        }

        public static Symbol Deconvolution(string symbolName,
                                           Symbol data,
                                           Symbol weight,
                                           Symbol bias,
                                           Shape kernel,
                                           uint32_t numFilter,
                                           Shape stride)
        {
            return Deconvolution(symbolName,
                                 data,
                                 weight,
                                 bias,
                                 kernel,
                                 numFilter,
                                 stride,
                                 new Shape());
        }

        public static Symbol Deconvolution(string symbolName,
                                           Symbol data,
                                           Symbol weight,
                                           Symbol bias,
                                           Shape kernel,
                                           uint32_t numFilter,
                                           Shape stride,
                                           Shape dilate)
        {
            return Deconvolution(symbolName,
                                 data,
                                 weight,
                                 bias,
                                 kernel,
                                 numFilter,
                                 stride,
                                 dilate,
                                 new Shape());
        }

        public static Symbol Deconvolution(string symbolName,
                                           Symbol data,
                                           Symbol weight,
                                           Symbol bias,
                                           Shape kernel,
                                           uint32_t numFilter,
                                           Shape stride,
                                           Shape dilate,
                                           Shape pad)
        {
            return Deconvolution(symbolName,
                                 data,
                                 weight,
                                 bias,
                                 kernel,
                                 numFilter,
                                 stride,
                                 dilate,
                                 pad,
                                 new Shape());
        }

        public static Symbol Deconvolution(string symbolName,
                                           Symbol data,
                                           Symbol weight,
                                           Symbol bias,
                                           Shape kernel,
                                           uint32_t numFilter,
                                           Shape stride,
                                           Shape dilate,
                                           Shape pad,
                                           Shape adj)
        {
            return Deconvolution(symbolName,
                                 data,
                                 weight,
                                 bias,
                                 kernel,
                                 numFilter,
                                 stride,
                                 dilate,
                                 pad,
                                 adj,
                                 new Shape());
        }

        public static Symbol Deconvolution(string symbolName,
                                           Symbol data,
                                           Symbol weight,
                                           Symbol bias,
                                           Shape kernel,
                                           uint32_t numFilter,
                                           Shape stride,
                                           Shape dilate,
                                           Shape pad,
                                           Shape adj,
                                           Shape targetShape,
                                           uint32_t numGroup = 1,
                                           uint64_t workspace = 512,
                                           bool noBias = true,
                                           DeconvolutionCudnnTune cudnnTune = DeconvolutionCudnnTune.None,
                                           bool cudnnOff = false,
                                           DeconvolutionLayout layout = DeconvolutionLayout.None)
        {
            return new Operator("Deconvolution").SetParam("kernel", kernel)
                                                .SetParam("num_filter", numFilter)
                                                .SetParam("stride", stride)
                                                .SetParam("dilate", dilate)
                                                .SetParam("pad", pad)
                                                .SetParam("adj", adj)
                                                .SetParam("target_shape", targetShape)
                                                .SetParam("num_group", numGroup)
                                                .SetParam("workspace", workspace)
                                                .SetParam("no_bias", noBias)
                                                .SetParam("cudnn_tune", DeconvolutionCudnnTuneValues[(int)cudnnTune])
                                                .SetParam("cudnn_off", cudnnOff)
                                                .SetParam("layout", DeconvolutionLayoutValues[(int)layout])
                                                .SetInput("data", data)
                                                .SetInput("weight", weight)
                                                .SetInput("bias", bias)
                                                .CreateSymbol(symbolName);
        }

        public static Symbol Deconvolution(Symbol data,
                                           Symbol weight,
                                           Symbol bias,
                                           Shape kernel,
                                           uint32_t numFilter)
        {
            return Deconvolution(data,
                                 weight,
                                 bias,
                                 kernel,
                                 numFilter,
                                 new Shape());
        }

        public static Symbol Deconvolution(Symbol data,
                                           Symbol weight,
                                           Symbol bias,
                                           Shape kernel,
                                           uint32_t numFilter,
                                           Shape stride)
        {
            return Deconvolution(data,
                                 weight,
                                 bias,
                                 kernel,
                                 numFilter,
                                 stride,
                                 new Shape());
        }

        public static Symbol Deconvolution(Symbol data,
                                           Symbol weight,
                                           Symbol bias,
                                           Shape kernel,
                                           uint32_t numFilter,
                                           Shape stride,
                                           Shape dilate)
        {
            return Deconvolution(data,
                                 weight,
                                 bias,
                                 kernel,
                                 numFilter,
                                 stride,
                                 dilate,
                                 new Shape());
        }

        public static Symbol Deconvolution(Symbol data,
                                           Symbol weight,
                                           Symbol bias,
                                           Shape kernel,
                                           uint32_t numFilter,
                                           Shape stride,
                                           Shape dilate,
                                           Shape pad)
        {
            return Deconvolution(data,
                                 weight,
                                 bias,
                                 kernel,
                                 numFilter,
                                 stride,
                                 dilate,
                                 pad,
                                 new Shape());
        }

        public static Symbol Deconvolution(Symbol data,
                                           Symbol weight,
                                           Symbol bias,
                                           Shape kernel,
                                           uint32_t numFilter,
                                           Shape stride,
                                           Shape dilate,
                                           Shape pad,
                                           Shape adj)
        {
            return Deconvolution(data,
                                 weight,
                                 bias,
                                 kernel,
                                 numFilter,
                                 stride,
                                 dilate,
                                 pad,
                                 adj,
                                 new Shape());
        }
        
        public static Symbol Deconvolution(Symbol data,
                            Symbol weight,
                            Symbol bias,
                            Shape kernel,
                            uint32_t numFilter,
                            Shape stride,
                            Shape dilate,
                            Shape pad,
                            Shape adj,
                            Shape targetShape,
                            uint32_t numGroup = 1,
                            uint64_t workspace = 512,
                            bool noBias = true,
                            DeconvolutionCudnnTune cudnnTune = DeconvolutionCudnnTune.None,
                            bool cudnnOff = false,
                            DeconvolutionLayout layout = DeconvolutionLayout.None)
        {
            return new Operator("Deconvolution").SetParam("kernel", kernel)
                                                .SetParam("num_filter", numFilter)
                                                .SetParam("stride", stride)
                                                .SetParam("dilate", dilate)
                                                .SetParam("pad", pad)
                                                .SetParam("adj", adj)
                                                .SetParam("target_shape", targetShape)
                                                .SetParam("num_group", numGroup)
                                                .SetParam("workspace", workspace)
                                                .SetParam("no_bias", noBias)
                                                .SetParam("cudnn_tune", DeconvolutionCudnnTuneValues[(int)cudnnTune])
                                                .SetParam("cudnn_off", cudnnOff)
                                                .SetParam("layout", DeconvolutionLayoutValues[(int)layout])
                                                .SetInput("data", data)
                                                .SetInput("weight", weight)
                                                .SetInput("bias", bias)
                                                .CreateSymbol();
        }

        #endregion

    }

}
