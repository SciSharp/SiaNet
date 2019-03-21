namespace SiaNet.Backend.MxNetLib
{

    public enum OpReqType
    {

        NullOp,

        WriteTo,

        WriteInplace,

        AddTo

    }

    /// <summary>
    /// Set the label that is reserved for blank label.If "first", 0-th label is reserved, and label values for tokens in the vocabulary are between ``1`` and ``alphabet_size-1``, and the padding mask is ``-1``. If "last", last label value ``alphabet_size-1`` is reserved for blank label instead, and label values for tokens in the vocabulary are between ``0`` and ``alphabet_size-2``, and the padding mask is ``0``.
    /// </summary>
    public enum ContribCtclossBlankLabel
    {
        First,
        Last
    };
    /// <summary>
    /// Output data type.
    /// </summary>
    public enum ContribDequantizeOutType
    {
        Float32
    };
    /// <summary>
    /// Output data type.
    /// </summary>
    public enum ContribQuantizeOutType
    {
        Uint8
    };
    /// <summary>
    /// Activation function to be applied.
    /// </summary>
    public enum LeakyreluActType
    {
        Elu,
        Leaky,
        Prelu,
        Rrelu
    };
    /// <summary>
    /// Padding type to use. "constant" pads with `constant_value` "edge" pads using the edge values of the input array "reflect" pads by reflecting values with respect to the edges.
    /// </summary>
    public enum PadMode
    {
        Constant,
        Edge,
        Reflect
    };
    /// <summary>
    /// Output storage type.
    /// </summary>
    public enum CastStorageStype
    {
        Csr,
        Default,
        RowSparse
    };
    /// <summary>
    /// Specify how out-of-bound indices bahave. "clip" means clip to the range. So, if all indices mentioned are too large, they are replaced by the index that addresses the last element along an axis.  "wrap" means to wrap around.  "raise" means to raise an error. 
    /// </summary>
    public enum TakeMode
    {
        Clip,
        Raise,
        Wrap
    };
    /// <summary>
    /// The return type. "value" means to return the top k values, "indices" means to return the indices of the top k values, "mask" means to return a mask array containing 0 and 1. 1 means the top k values. "both" means to return a list of both values and indices of top k elements.
    /// </summary>
    public enum TopkRetTyp
    {
        Both,
        Indices,
        Mask,
        Value
    };
    /// <summary>
    /// upsampling method
    /// </summary>
    public enum UpsamplingSampleType
    {
        Bilinear,
        Nearest
    };
    /// <summary>
    /// How to handle multiple input. concat means concatenate upsampled images along the channel dimension. sum means add all images together, only available for nearest neighbor upsampling.
    /// </summary>
    public enum UpsamplingMultiInputMode
    {
        Concat,
        Sum
    };
   
    /// <summary>
    /// Set layout for input, output and weight. Empty for    default layout: NCW for 1d, NCHW for 2d and NCDHW for 3d.
    /// </summary>
    public enum ContribDeformableconvolutionLayout
    {
        NCDHW,
        NCHW,
        NCW
    };
    
    /// <summary>
    /// Whether to pick convolution algo by running performance test.    Leads to higher startup time but may give faster speed. Options are:    'off': no tuning    'limited_workspace': run test and pick the fastest algorithm that doesn't exceed workspace limit.    'fastest': pick the fastest algorithm and ignore workspace limit.    If set to None (default), behavior is determined by environment    variable MXNET_CUDNN_AUTOTUNE_DEFAULT: 0 for off,    1 for limited workspace (default), 2 for fastest.
    /// </summary>
    public enum ConvolutionV1CudnnTune
    {
        Fastest,
        LimitedWorkspace,
        Off
    };
    /// <summary>
    /// Set layout for input, output and weight. Empty for    default layout: NCHW for 2d and NCDHW for 3d.
    /// </summary>
    public enum ConvolutionV1Layout
    {
        NCDHW,
        NCHW,
        NDHWC,
        NHWC
    };
   
    /// <summary>
    /// The type of transformation. For `affine`, input data should be an affine matrix of size (batch, 6). For `warp`, input data should be an optical flow of size (batch, 2, h, w).
    /// </summary>
    public enum GridgeneratorTransformType
    {
        Affine,
        Warp
    };
    /// <summary>
    /// Specify the dimension along which to compute L2 norm.
    /// </summary>
    public enum L2normalizationMode
    {
        Channel,
        Instance,
        Spatial
    };
    /// <summary>
    /// If this is set to null, the output gradient will not be normalized. If this is set to batch, the output gradient will be divided by the batch size. If this is set to valid, the output gradient will be divided by the number of valid input elements.
    /// </summary>
    public enum MakelossNormalization
    {
        Batch,
        Null,
        Valid
    };
   
    /// <summary>
    /// Pooling type to be applied.
    /// </summary>
    public enum PoolingV1PoolType
    {
        Avg,
        Max,
        Sum
    };
    /// <summary>
    /// Pooling convention to be applied.
    /// </summary>
    public enum PoolingV1PoolingConvention
    {
        Full,
        Valid
    };
   
    /// <summary>
    /// Specifies how to compute the softmax. If set to ``instance``, it computes softmax for each instance. If set to ``channel``, It computes cross channel softmax for each position of each instance.
    /// </summary>
    public enum SoftmaxactivationMode
    {
        Channel,
        Instance
    };
    /// <summary>
    /// Normalizes the gradient.
    /// </summary>
    public enum SoftmaxoutputNormalization
    {
        Batch,
        Null,
        Valid
    };
    /// <summary>
    /// Normalizes the gradient.
    /// </summary>
    public enum SoftmaxNormalization
    {
        Batch,
        Null,
        Valid
    };
    /// <summary>
    /// transformation type
    /// </summary>
    public enum SpatialtransformerTransformType
    {
        Affine
    };
    /// <summary>
    /// sampling type
    /// </summary>
    public enum SpatialtransformerSamplerType
    {
        Bilinear
    };


}

