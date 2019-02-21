// ***********************************************************************
// Assembly         : SiaNet.Backend.TensorSharp.CUDA91
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="DNN.cs" company="SiaNet.Backend.TensorSharp.CUDA91">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using ManagedCuda;
using ManagedCuda.CudaDNN;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SiaNet.Backend.TensorSharp.Core;

namespace SiaNet.Backend.TensorSharp.CUDA
{
    /// <summary>
    /// Class TensorShape.
    /// </summary>
    public class TensorShape
    {
        /// <summary>
        /// Gets the type of the element.
        /// </summary>
        /// <value>The type of the element.</value>
        public DType ElementType { get; private set; }
        /// <summary>
        /// Gets the sizes.
        /// </summary>
        /// <value>The sizes.</value>
        public long[] Sizes { get; private set; }
        /// <summary>
        /// Gets the strides.
        /// </summary>
        /// <value>The strides.</value>
        public long[] Strides { get; private set; }
        /// <summary>
        /// Gets the dimension count.
        /// </summary>
        /// <value>The dimension count.</value>
        public int DimensionCount { get { return Sizes.Length; } }

        /// <summary>
        /// Initializes a new instance of the <see cref="TensorShape"/> class.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        public TensorShape(NDArray tensor)
        {
            this.ElementType = tensor.ElementType;
            this.Sizes = tensor.Shape;
            this.Strides = tensor.Strides;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="TensorShape"/> class.
        /// </summary>
        /// <param name="elementType">Type of the element.</param>
        /// <param name="sizes">The sizes.</param>
        /// <param name="strides">The strides.</param>
        public TensorShape(DType elementType, long[] sizes, long[] strides)
        {
            this.ElementType = elementType;
            this.Sizes = sizes;
            this.Strides = strides;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="TensorShape"/> class.
        /// </summary>
        /// <param name="elementType">Type of the element.</param>
        /// <param name="sizes">The sizes.</param>
        public TensorShape(DType elementType, long[] sizes)
            : this(elementType, sizes, TensorDimensionHelpers.GetContiguousStride(sizes))
        {
        }
    }

    /// <summary>
    /// Enum DNNActivation
    /// </summary>
    public enum DNNActivation
    {
        /// <summary>
        /// The sigmoid
        /// </summary>
        Sigmoid = cudnnActivationMode.Sigmoid,
        /// <summary>
        /// The relu
        /// </summary>
        Relu = cudnnActivationMode.Relu,
        /// <summary>
        /// The tanh
        /// </summary>
        Tanh = cudnnActivationMode.Tanh,
        /// <summary>
        /// The clipped relu
        /// </summary>
        ClippedRelu = cudnnActivationMode.ClippedRelu,
    }

    /// <summary>
    /// Enum DNNSoftmaxAlgorithm
    /// </summary>
    public enum DNNSoftmaxAlgorithm
    {
        /// <summary>
        /// The fast
        /// </summary>
        Fast = cudnnSoftmaxAlgorithm.Fast,
        /// <summary>
        /// The accurate
        /// </summary>
        Accurate = cudnnSoftmaxAlgorithm.Accurate,
        /// <summary>
        /// The log
        /// </summary>
        Log = cudnnSoftmaxAlgorithm.Log,
    }

    /// <summary>
    /// Enum DNNSoftmaxMode
    /// </summary>
    public enum DNNSoftmaxMode
    {
        /// <summary>
        /// The instance
        /// </summary>
        Instance = cudnnSoftmaxMode.Instance,
        /// <summary>
        /// The channel
        /// </summary>
        Channel = cudnnSoftmaxMode.Channel,
    }

    /// <summary>
    /// Enum DNNPoolingMode
    /// </summary>
    public enum DNNPoolingMode
    {
        /// <summary>
        /// Determines the maximun of the parameters.
        /// </summary>
        Max = cudnnPoolingMode.Max,
        /// <summary>
        /// The average count include padding
        /// </summary>
        AverageCountIncludePadding = cudnnPoolingMode.AverageCountIncludePadding,
        /// <summary>
        /// The average count exclude padding
        /// </summary>
        AverageCountExcludePadding = cudnnPoolingMode.AverageCountExcludePadding,
    }

    /// <summary>
    /// Enum DNNConvolutionFwdAlgo
    /// </summary>
    public enum DNNConvolutionFwdAlgo
    {
        /// <summary>
        /// The implicit gemm
        /// </summary>
        ImplicitGEMM = cudnnConvolutionFwdAlgo.ImplicitGEMM,
        /// <summary>
        /// The implicit precomp gemm
        /// </summary>
        ImplicitPrecompGEMM = cudnnConvolutionFwdAlgo.ImplicitPrecompGEMM,
        /// <summary>
        /// The gemm
        /// </summary>
        GEMM = cudnnConvolutionFwdAlgo.GEMM,
        /// <summary>
        /// The direct
        /// </summary>
        Direct = cudnnConvolutionFwdAlgo.Direct,
        /// <summary>
        /// The FFT
        /// </summary>
        FFT = cudnnConvolutionFwdAlgo.FFT,
        /// <summary>
        /// The FFT with tiling
        /// </summary>
        FFTWithTiling = cudnnConvolutionFwdAlgo.FFTWithTiling,
        /// <summary>
        /// The winograd
        /// </summary>
        Winograd = cudnnConvolutionFwdAlgo.Winograd,
    }

    /// <summary>
    /// Enum DNNConvolutionBwdFilterAlgo
    /// </summary>
    public enum DNNConvolutionBwdFilterAlgo
    {
        /// <summary>
        /// The algo0
        /// </summary>
        Algo0 = cudnnConvolutionBwdFilterAlgo.Algo0,
        /// <summary>
        /// The algo1
        /// </summary>
        Algo1 = cudnnConvolutionBwdFilterAlgo.Algo1,
        /// <summary>
        /// The algo3
        /// </summary>
        Algo3 = cudnnConvolutionBwdFilterAlgo.Algo3,
        /// <summary>
        /// The algo FFT
        /// </summary>
        AlgoFFT = cudnnConvolutionBwdFilterAlgo.AlgoFFT,
    }

    /// <summary>
    /// Enum DNNConvolutionBwdDataAlgo
    /// </summary>
    public enum DNNConvolutionBwdDataAlgo
    {
        /// <summary>
        /// The algo0
        /// </summary>
        Algo0 = cudnnConvolutionBwdDataAlgo.Algo0,
        /// <summary>
        /// The algo1
        /// </summary>
        Algo1 = cudnnConvolutionBwdDataAlgo.Algo1,
        /// <summary>
        /// The algo FFT
        /// </summary>
        AlgoFFT = cudnnConvolutionBwdDataAlgo.AlgoFFT,
        /// <summary>
        /// The winograd
        /// </summary>
        Winograd = cudnnConvolutionBwdDataAlgo.Winograd,
    }


    /// <summary>
    /// Struct DNNPoolingDesc
    /// </summary>
    public struct DNNPoolingDesc
    {
        /// <summary>
        /// The mode
        /// </summary>
        public DNNPoolingMode Mode;
        /// <summary>
        /// The window dims
        /// </summary>
        public int[] WindowDims;
        /// <summary>
        /// The padding
        /// </summary>
        public int[] Padding;
        /// <summary>
        /// The strides
        /// </summary>
        public int[] Strides;

        /// <summary>
        /// Initializes a new instance of the <see cref="DNNPoolingDesc"/> struct.
        /// </summary>
        /// <param name="mode">The mode.</param>
        /// <param name="windowDims">The window dims.</param>
        /// <param name="padding">The padding.</param>
        /// <param name="strides">The strides.</param>
        public DNNPoolingDesc(DNNPoolingMode mode, int[] windowDims, int[] padding, int[] strides)
        {
            this.Mode = mode;
            this.WindowDims = windowDims;
            this.Padding = padding;
            this.Strides = strides;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="DNNPoolingDesc"/> struct.
        /// </summary>
        /// <param name="mode">The mode.</param>
        /// <param name="dimA">The dim a.</param>
        /// <param name="dimB">The dim b.</param>
        /// <param name="padA">The pad a.</param>
        /// <param name="padB">The pad b.</param>
        /// <param name="strideA">The stride a.</param>
        /// <param name="strideB">The stride b.</param>
        public DNNPoolingDesc(DNNPoolingMode mode, int dimA, int dimB, int padA, int padB, int strideA, int strideB)
        {
            this.Mode = mode;
            this.WindowDims = new int[] { dimA, dimB };
            this.Padding = new int[] { padA, padB };
            this.Strides = new int[] { strideA, strideB };
        }
    }


    /// <summary>
    /// Class DNN.
    /// </summary>
    public static class DNN
    {
        /// <summary>
        /// Activations the forward.
        /// </summary>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <param name="activationType">Type of the activation.</param>
        /// <param name="clippedReluCeiling">The clipped relu ceiling.</param>
        public static void ActivationForward(NDArray x, NDArray y, DNNActivation activationType, double clippedReluCeiling)
        {
            using (var dnn = CudaHelpers.TSContextForTensor(x).DNNForTensor(x))
            {

                var activationDesc = new ActivationDescriptor();
                activationDesc.SetActivationDescriptor((cudnnActivationMode)activationType,
                    cudnnNanPropagation.PropagateNan,
                    clippedReluCeiling);

                using (var xPtr = GetDeviceVar(x))
                using (var yPtr = GetDeviceVar(y))
                using (var xDesc = GetDescriptor(x))
                using (var yDesc = GetDescriptor(y))
                {
                    dnn.Value.ActivationForward(activationDesc, 1,
                        xDesc, xPtr,
                        0,
                        yDesc, yPtr);
                }
            }
        }

        /// <summary>
        /// Activations the backward.
        /// </summary>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <param name="dx">The dx.</param>
        /// <param name="dy">The dy.</param>
        /// <param name="activationType">Type of the activation.</param>
        /// <param name="clippedReluCeiling">The clipped relu ceiling.</param>
        public static void ActivationBackward(NDArray x, NDArray y, NDArray dx, NDArray dy, DNNActivation activationType, double clippedReluCeiling)
        {
            using (var dnn = CudaHelpers.TSContextForTensor(x).DNNForTensor(x))
            {

                var activationDesc = new ActivationDescriptor();
                activationDesc.SetActivationDescriptor((cudnnActivationMode)activationType,
                    cudnnNanPropagation.PropagateNan,
                    clippedReluCeiling);

                using (var xPtr = GetDeviceVar(x))
                using (var yPtr = GetDeviceVar(y))
                using (var dxPtr = GetDeviceVar(dx))
                using (var dyPtr = GetDeviceVar(dy))
                using (var xDesc = GetDescriptor(x))
                using (var yDesc = GetDescriptor(y))
                using (var dxDesc = GetDescriptor(dx))
                using (var dyDesc = GetDescriptor(dy))
                {
                    dnn.Value.ActivationBackward(activationDesc, 1,
                        xDesc, xPtr,
                        dxDesc, dxPtr,
                        yDesc, yPtr,
                        0,
                        dyDesc, dyPtr);
                }
            }
        }

        /// <summary>
        /// Softmaxes the forward.
        /// </summary>
        /// <param name="algorithm">The algorithm.</param>
        /// <param name="mode">The mode.</param>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        public static void SoftmaxForward(DNNSoftmaxAlgorithm algorithm, DNNSoftmaxMode mode, NDArray x, NDArray y)
        {
            using (var dnn = CudaHelpers.TSContextForTensor(x).DNNForTensor(x))
            {

                using (var xPtr = GetDeviceVar(x))
                using (var yPtr = GetDeviceVar(y))
                using (var xDesc = GetDescriptor(x))
                using (var yDesc = GetDescriptor(y))
                {
                    dnn.Value.SoftmaxForward((cudnnSoftmaxAlgorithm)algorithm, (cudnnSoftmaxMode)mode, 1,
                    xDesc, xPtr,
                    0,
                    yDesc, yPtr);
                }
            }
        }

        /// <summary>
        /// Softmaxes the backward.
        /// </summary>
        /// <param name="algorithm">The algorithm.</param>
        /// <param name="mode">The mode.</param>
        /// <param name="y">The y.</param>
        /// <param name="dx">The dx.</param>
        /// <param name="dy">The dy.</param>
        public static void SoftmaxBackward(DNNSoftmaxAlgorithm algorithm, DNNSoftmaxMode mode, NDArray y, NDArray dx, NDArray dy)
        {
            using (var dnn = CudaHelpers.TSContextForTensor(y).DNNForTensor(y))
            {

                using (var yPtr = GetDeviceVar(y))
                using (var dxPtr = GetDeviceVar(dx))
                using (var dyPtr = GetDeviceVar(dy))
                using (var yDesc = GetDescriptor(y))
                using (var dxDesc = GetDescriptor(dx))
                using (var dyDesc = GetDescriptor(dy))
                {
                    dnn.Value.SoftmaxBackward((cudnnSoftmaxAlgorithm)algorithm, (cudnnSoftmaxMode)mode, 1,
                    yDesc, yPtr,
                    dyDesc, dyPtr,
                    0,
                    dxDesc, dxPtr);
                }
            }
        }

        /// <summary>
        /// Poolings the forward.
        /// </summary>
        /// <param name="desc">The desc.</param>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        public static void PoolingForward(DNNPoolingDesc desc, NDArray x, NDArray y)
        {
            using (var dnn = CudaHelpers.TSContextForTensor(x).DNNForTensor(x))
            {

                var poolingDesc = new PoolingDescriptor();
                poolingDesc.SetPoolingNdDescriptor((cudnnPoolingMode)desc.Mode, cudnnNanPropagation.PropagateNan, desc.WindowDims.Length,
                    desc.WindowDims, desc.Padding, desc.Strides);

                using (var xPtr = GetDeviceVar(x))
                using (var yPtr = GetDeviceVar(y))
                using (var xDesc = GetDescriptor(x))
                using (var yDesc = GetDescriptor(y))
                {
                    dnn.Value.PoolingForward(poolingDesc, 1,
                        xDesc, xPtr,
                        0,
                        yDesc, yPtr);
                }
            }
        }

        /// <summary>
        /// Poolings the backward.
        /// </summary>
        /// <param name="desc">The desc.</param>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <param name="dx">The dx.</param>
        /// <param name="dy">The dy.</param>
        public static void PoolingBackward(DNNPoolingDesc desc, NDArray x, NDArray y, NDArray dx, NDArray dy)
        {
            using (var dnn = CudaHelpers.TSContextForTensor(x).DNNForTensor(x))
            {
                var poolingDesc = new PoolingDescriptor();
                poolingDesc.SetPoolingNdDescriptor((cudnnPoolingMode)desc.Mode, cudnnNanPropagation.PropagateNan, desc.WindowDims.Length,
                    desc.WindowDims, desc.Padding, desc.Strides);

                using (var xPtr = GetDeviceVar(x))
                using (var yPtr = GetDeviceVar(y))
                using (var dxPtr = GetDeviceVar(dx))
                using (var dyPtr = GetDeviceVar(dy))
                using (var xDesc = GetDescriptor(x))
                using (var yDesc = GetDescriptor(y))
                using (var dxDesc = GetDescriptor(dx))
                using (var dyDesc = GetDescriptor(dy))
                {
                    // Note: ManagedCUDA argument names may be slightly misleading (src refers to 'y' here, and dest to 'x')
                    dnn.Value.PoolingBackward(poolingDesc, 1,
                        yDesc, yPtr,
                        dyDesc, dyPtr,
                        xDesc, xPtr,
                        0,
                        dxDesc, dxPtr);
                }
            }
        }



        /// <summary>
        /// Adds the tensor.
        /// </summary>
        /// <param name="src">The source.</param>
        /// <param name="result">The result.</param>
        public static void AddTensor(NDArray src, NDArray result)
        {
            using (var dnn = CudaHelpers.TSContextForTensor(src).DNNForTensor(src))
            {

                using (var srcPtr = GetDeviceVar(src))
                using (var resultPtr = GetDeviceVar(result))
                using (var srcDesc = GetDescriptor(src))
                using (var resultDesc = GetDescriptor(result))
                {
                    dnn.Value.AddTensor(1,
                        srcDesc, srcPtr,
                        1,
                        resultDesc, resultPtr);
                }
            }
        }

        /// <summary>
        /// Gets the conv descriptor.
        /// </summary>
        /// <param name="cd">The cd.</param>
        /// <param name="elementType">Type of the element.</param>
        /// <returns>ConvolutionDescriptor.</returns>
        private static ConvolutionDescriptor GetConvDescriptor(Cpu.ConvolutionDesc2d cd, DType elementType)
        {
            var convDesc = new ConvolutionDescriptor();
            convDesc.SetConvolution2dDescriptor(cd.padH, cd.padW, cd.dH, cd.dW, 1, 1, cudnnConvolutionMode.CrossCorrelation, GetDataType(elementType));
            return convDesc;
        }

        /// <summary>
        /// Convs the forward.
        /// </summary>
        /// <param name="algo">The algo.</param>
        /// <param name="cd">The cd.</param>
        /// <param name="workspace">The workspace.</param>
        /// <param name="x">The x.</param>
        /// <param name="w">The w.</param>
        /// <param name="y">The y.</param>
        public static void ConvForward(DNNConvolutionFwdAlgo algo, Cpu.ConvolutionDesc2d cd, CudaStorage workspace, NDArray x, NDArray w, NDArray y)
        {
            using (var dnn = CudaHelpers.TSContextForTensor(x).DNNForTensor(x))
            {
                var convDesc = GetConvDescriptor(cd, x.ElementType);

                using (var workspacePtr = new CudaDeviceVariable<byte>(workspace.DevicePtrAtElement(0), false, workspace.ByteLength))
                using (var xPtr = GetDeviceVar(x))
                using (var wPtr = GetDeviceVar(w))
                using (var yPtr = GetDeviceVar(y))
                using (var xDesc = GetDescriptor(x))
                using (var wDesc = GetFilterDescriptor(w))
                using (var yDesc = GetDescriptor(y))
                {
                    dnn.Value.ConvolutionForward(1,
                        xDesc, xPtr,
                        wDesc, wPtr,
                        convDesc,
                        (cudnnConvolutionFwdAlgo)algo,
                        workspacePtr,
                        0,
                        yDesc, yPtr);
                }
            }
        }

        /// <summary>
        /// Convolutions the backward data.
        /// </summary>
        /// <param name="algo">The algo.</param>
        /// <param name="cd">The cd.</param>
        /// <param name="workspace">The workspace.</param>
        /// <param name="w">The w.</param>
        /// <param name="dy">The dy.</param>
        /// <param name="dx">The dx.</param>
        public static void ConvolutionBackwardData(DNNConvolutionBwdDataAlgo algo, Cpu.ConvolutionDesc2d cd, CudaStorage workspace, NDArray w, NDArray dy, NDArray dx)
        {
            using (var dnn = CudaHelpers.TSContextForTensor(w).DNNForTensor(w))
            {

                var convDesc = GetConvDescriptor(cd, w.ElementType);

                using (var workspacePtr = new CudaDeviceVariable<byte>(workspace.DevicePtrAtElement(0), false, workspace.ByteLength))
                using (var wPtr = GetDeviceVar(w))
                using (var dxPtr = GetDeviceVar(dx))
                using (var dyPtr = GetDeviceVar(dy))
                using (var wDesc = GetFilterDescriptor(w))
                using (var dxDesc = GetDescriptor(dx))
                using (var dyDesc = GetDescriptor(dy))
                {
                    dnn.Value.ConvolutionBackwardData(1,
                        wDesc, wPtr,
                        dyDesc, dyPtr,
                        convDesc,
                        (cudnnConvolutionBwdDataAlgo)algo,
                        workspacePtr, 0f,
                        dxDesc, dxPtr);
                }
            }
        }

        /// <summary>
        /// Convolutions the backward filter.
        /// </summary>
        /// <param name="algo">The algo.</param>
        /// <param name="cd">The cd.</param>
        /// <param name="workspace">The workspace.</param>
        /// <param name="x">The x.</param>
        /// <param name="dy">The dy.</param>
        /// <param name="dw">The dw.</param>
        public static void ConvolutionBackwardFilter(DNNConvolutionBwdFilterAlgo algo, Cpu.ConvolutionDesc2d cd, CudaStorage workspace, NDArray x, NDArray dy, NDArray dw)
        {
            using (var dnn = CudaHelpers.TSContextForTensor(x).DNNForTensor(x))
            {
                var convDesc = GetConvDescriptor(cd, x.ElementType);

                using (var workspacePtr = new CudaDeviceVariable<byte>(workspace.DevicePtrAtElement(0), false, workspace.ByteLength))
                using (var xPtr = GetDeviceVar(x))
                using (var dyPtr = GetDeviceVar(dy))
                using (var dwPtr = GetDeviceVar(dw))
                using (var xDesc = GetDescriptor(x))
                using (var dyDesc = GetDescriptor(dy))
                using (var dwDesc = GetFilterDescriptor(dw))
                {
                    dnn.Value.ConvolutionBackwardFilter(1,
                        xDesc, xPtr,
                        dyDesc, dyPtr,
                        convDesc,
                        (cudnnConvolutionBwdFilterAlgo)algo,
                        workspacePtr,
                        0,
                        dwDesc, dwPtr);
                }
            }
        }


        /// <summary>
        /// Convolutions the backward bias.
        /// </summary>
        /// <param name="cd">The cd.</param>
        /// <param name="dy">The dy.</param>
        /// <param name="db">The database.</param>
        public static void ConvolutionBackwardBias(Cpu.ConvolutionDesc2d cd, NDArray dy, NDArray db)
        {
            using (var dnn = CudaHelpers.TSContextForTensor(dy).DNNForTensor(dy))
            {

                using (var dyPtr = GetDeviceVar(dy))
                using (var dbPtr = GetDeviceVar(db))
                using (var dyDesc = GetDescriptor(dy))
                using (var dbDesc = GetDescriptor(db))
                {
                    dnn.Value.ConvolutionBackwardBias(1,
                        dyDesc, dyPtr,
                        0,
                        dbDesc, dbPtr);
                }
            }
        }


        /// <summary>
        /// Gets the size of the convolution forward workspace.
        /// </summary>
        /// <param name="allocator">The allocator.</param>
        /// <param name="algo">The algo.</param>
        /// <param name="cd">The cd.</param>
        /// <param name="x">The x.</param>
        /// <param name="w">The w.</param>
        /// <param name="y">The y.</param>
        /// <returns>System.Int64.</returns>
        /// <exception cref="InvalidOperationException">allocator must be a CUDA allocator</exception>
        public static long GetConvolutionForwardWorkspaceSize(IAllocator allocator, DNNConvolutionFwdAlgo algo, Cpu.ConvolutionDesc2d cd, TensorShape x, TensorShape w, TensorShape y)
        {
            if (!(allocator is CudaAllocator))
                throw new InvalidOperationException("allocator must be a CUDA allocator");

            var cudaAllocator = (CudaAllocator)allocator;

            using (var dnn = cudaAllocator.Context.DNNForDevice(cudaAllocator.DeviceId))
            {

                var convDesc = GetConvDescriptor(cd, x.ElementType);

                using (var xDesc = GetDescriptor(x))
                using (var wDesc = GetFilterDescriptor(w))
                using (var yDesc = GetDescriptor(y))
                {
                    return dnn.Value.GetConvolutionForwardWorkspaceSize(
                        xDesc,
                        wDesc,
                        convDesc,
                        yDesc,
                        (cudnnConvolutionFwdAlgo)algo);
                }
            }
        }

        /// <summary>
        /// Gets the size of the convolution backward filter workspace.
        /// </summary>
        /// <param name="allocator">The allocator.</param>
        /// <param name="algo">The algo.</param>
        /// <param name="cd">The cd.</param>
        /// <param name="x">The x.</param>
        /// <param name="dy">The dy.</param>
        /// <param name="dw">The dw.</param>
        /// <returns>System.Int64.</returns>
        /// <exception cref="InvalidOperationException">allocator must be a CUDA allocator</exception>
        public static long GetConvolutionBackwardFilterWorkspaceSize(IAllocator allocator, DNNConvolutionBwdFilterAlgo algo, Cpu.ConvolutionDesc2d cd, TensorShape x, TensorShape dy, TensorShape dw)
        {
            if (!(allocator is CudaAllocator))
                throw new InvalidOperationException("allocator must be a CUDA allocator");

            var cudaAllocator = (CudaAllocator)allocator;

            using (var dnn = cudaAllocator.Context.DNNForDevice(cudaAllocator.DeviceId))
            {

                var convDesc = GetConvDescriptor(cd, x.ElementType);

                using (var xDesc = GetDescriptor(x))
                using (var dyDesc = GetDescriptor(dy))
                using (var dwDesc = GetFilterDescriptor(dw))
                {
                    return dnn.Value.GetConvolutionBackwardFilterWorkspaceSize(
                        xDesc,
                        dyDesc,
                        convDesc,
                        dwDesc,
                        (cudnnConvolutionBwdFilterAlgo)algo);
                }
            }
        }

        /// <summary>
        /// Gets the size of the convolution backward data workspace.
        /// </summary>
        /// <param name="allocator">The allocator.</param>
        /// <param name="algo">The algo.</param>
        /// <param name="cd">The cd.</param>
        /// <param name="w">The w.</param>
        /// <param name="dy">The dy.</param>
        /// <param name="dx">The dx.</param>
        /// <returns>System.Int64.</returns>
        /// <exception cref="InvalidOperationException">allocator must be a CUDA allocator</exception>
        public static long GetConvolutionBackwardDataWorkspaceSize(IAllocator allocator, DNNConvolutionBwdDataAlgo algo, Cpu.ConvolutionDesc2d cd, TensorShape w, TensorShape dy, TensorShape dx)
        {
            if (!(allocator is CudaAllocator))
                throw new InvalidOperationException("allocator must be a CUDA allocator");

            var cudaAllocator = (CudaAllocator)allocator;

            using (var dnn = cudaAllocator.Context.DNNForDevice(cudaAllocator.DeviceId))
            {

                var convDesc = GetConvDescriptor(cd, w.ElementType);

                using (var wDesc = GetFilterDescriptor(w))
                using (var dyDesc = GetDescriptor(dy))
                using (var dxDesc = GetDescriptor(dx))
                {
                    return dnn.Value.GetConvolutionBackwardDataWorkspaceSize(
                        wDesc,
                        dyDesc,
                        convDesc,
                        dxDesc,
                        (cudnnConvolutionBwdDataAlgo)algo);
                }
            }
        }


        /// <summary>
        /// Gets the device variable.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <returns>CudaDeviceVariable&lt;System.Single&gt;.</returns>
        private static CudaDeviceVariable<float> GetDeviceVar(NDArray tensor)
        {
            var ptr = CudaHelpers.GetBufferStart(tensor);
            return new CudaDeviceVariable<float>(ptr, false, 0);// set size to 0 because we never end up using it
        }

        /// <summary>
        /// Gets the descriptor.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <returns>TensorDescriptor.</returns>
        private static TensorDescriptor GetDescriptor(NDArray tensor)
        {
            var result = new TensorDescriptor();
            result.SetTensorNdDescriptor(
                GetDataType(tensor.ElementType),
                tensor.DimensionCount,
                tensor.Shape.Select(x => (int)x).ToArray(),
                tensor.Strides.Select(x => (int)x).ToArray());
            return result;
        }

        /// <summary>
        /// Gets the filter descriptor.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <returns>FilterDescriptor.</returns>
        private static FilterDescriptor GetFilterDescriptor(NDArray tensor)
        {
            var result = new FilterDescriptor();
            result.SetFilterNdDescriptor(
                GetDataType(tensor.ElementType),
                cudnnTensorFormat.NCHW,
                tensor.DimensionCount,
                tensor.Shape.Select(x => (int)x).ToArray());
            return result;
        }

        /// <summary>
        /// Gets the descriptor.
        /// </summary>
        /// <param name="shape">The shape.</param>
        /// <returns>TensorDescriptor.</returns>
        private static TensorDescriptor GetDescriptor(TensorShape shape)
        {
            var result = new TensorDescriptor();
            result.SetTensorNdDescriptor(
                GetDataType(shape.ElementType),
                shape.DimensionCount,
                shape.Sizes.Select(x => (int)x).ToArray(),
                shape.Strides.Select(x => (int)x).ToArray());
            return result;
        }

        /// <summary>
        /// Gets the filter descriptor.
        /// </summary>
        /// <param name="shape">The shape.</param>
        /// <returns>FilterDescriptor.</returns>
        private static FilterDescriptor GetFilterDescriptor(TensorShape shape)
        {
            var result = new FilterDescriptor();
            result.SetFilterNdDescriptor(
                GetDataType(shape.ElementType),
                cudnnTensorFormat.NCHW,
                shape.DimensionCount,
                shape.Sizes.Select(x => (int)x).ToArray());
            return result;
        }


        /// <summary>
        /// Gets the type of the data.
        /// </summary>
        /// <param name="dataType">Type of the data.</param>
        /// <returns>cudnnDataType.</returns>
        /// <exception cref="NotSupportedException">DNN: type not supported: " + dataType</exception>
        private static cudnnDataType GetDataType(DType dataType)
        {
            switch(dataType)
            {
                case DType.Float32: return cudnnDataType.Float;
                case DType.Float64: return cudnnDataType.Double;
                case DType.Float16: return cudnnDataType.Half;
                default:
                    throw new NotSupportedException("DNN: type not supported: " + dataType);
            }
        }
    }
}
