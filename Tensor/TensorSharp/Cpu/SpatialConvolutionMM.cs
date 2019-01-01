// ***********************************************************************
// Assembly         : TensorSharp
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="SpatialConvolutionMM.cs" company="TensorSharp">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace TensorSharp.Cpu
{
    /// <summary>
    /// Class ConvolutionDesc2d.
    /// </summary>
    public class ConvolutionDesc2d
    {
        /// <summary>
        /// The k w
        /// </summary>
        public int kW;
        /// <summary>
        /// The k h
        /// </summary>
        public int kH;
        /// <summary>
        /// The d w
        /// </summary>
        public int dW;
        /// <summary>
        /// The d h
        /// </summary>
        public int dH;
        /// <summary>
        /// The pad w
        /// </summary>
        public int padW;
        /// <summary>
        /// The pad h
        /// </summary>
        public int padH;

        /// <summary>
        /// Initializes a new instance of the <see cref="ConvolutionDesc2d"/> class.
        /// </summary>
        /// <param name="kW">The k w.</param>
        /// <param name="kH">The k h.</param>
        /// <param name="dW">The d w.</param>
        /// <param name="dH">The d h.</param>
        /// <param name="padW">The pad w.</param>
        /// <param name="padH">The pad h.</param>
        public ConvolutionDesc2d(int kW, int kH, int dW, int dH, int padW, int padH)
        {
            this.kW = kW;
            this.kH = kH;
            this.dW = dW;
            this.dH = dH;
            this.padW = padW;
            this.padH = padH;
        }
    }

    /// <summary>
    /// Class SpatialConvolutionMM.
    /// </summary>
    public static class SpatialConvolutionMM
    {
        /// <summary>
        /// Outputs the size.
        /// </summary>
        /// <param name="inputSizes">The input sizes.</param>
        /// <param name="weightSizes">The weight sizes.</param>
        /// <param name="cd">The cd.</param>
        /// <returns>System.Int64[].</returns>
        public static long[] OutputSize(long[] inputSizes, long[] weightSizes, ConvolutionDesc2d cd)
        {
            int dimf = 1;
            int dimw = 3;
            int dimh = 2;

            var n = inputSizes[0];
            var nInputPlane = inputSizes[dimf];
            var inputWidth = inputSizes[dimw];
            var inputHeight = inputSizes[dimh];
            var nOutputPlane = weightSizes[0];

            var outputWidth = (inputWidth + 2 * cd.padW - cd.kW) / cd.dW + 1;
            var outputHeight = (inputHeight + 2 * cd.padH - cd.kH) / cd.dH + 1;

            return new long[] { n, nOutputPlane, outputHeight, outputWidth };
        }

        /// <summary>
        /// fs the size of the input.
        /// </summary>
        /// <param name="inputSizes">The input sizes.</param>
        /// <param name="outputSizes">The output sizes.</param>
        /// <param name="cd">The cd.</param>
        /// <returns>System.Int64[].</returns>
        public static long[] FInputSize(long[] inputSizes, long[] outputSizes, ConvolutionDesc2d cd)
        {
            return new long[] { inputSizes[0], cd.kW * cd.kH * inputSizes[1], outputSizes[2] * outputSizes[3] };
        }


        /// <summary>
        /// Conv2s the forward.
        /// </summary>
        /// <param name="input">The input.</param>
        /// <param name="output">The output.</param>
        /// <param name="weight">The weight.</param>
        /// <param name="bias">The bias.</param>
        /// <param name="finput">The finput.</param>
        /// <param name="cd">The cd.</param>
        /// <exception cref="InvalidOperationException">
        /// bias has incorrect size. Expected 1D tensor of size " + nOutputPlane
        /// or
        /// or
        /// or
        /// 4D input expected (NCHW order)
        /// or
        /// finput is incorrect size
        /// or
        /// output is incorrect size
        /// </exception>
        public static void Conv2Forward(Tensor input, Tensor output, Tensor weight, Tensor bias, Tensor finput, ConvolutionDesc2d cd)
        {
            int dimf = 1;
            int dimw = 3;
            int dimh = 2;

            var n = input.Shape[0];
            var nInputPlane = input.Shape[dimf];
            var inputWidth = input.Shape[dimw];
            var inputHeight = input.Shape[dimh];
            var nOutputPlane = weight.Shape[0];

            var outputWidth = (inputWidth + 2 * cd.padW - cd.kW) / cd.dW + 1;
            var outputHeight = (inputHeight + 2 * cd.padH - cd.kH) / cd.dH + 1;

            if (bias != null && (bias.Shape[0] != nOutputPlane))
                throw new InvalidOperationException("bias has incorrect size. Expected 1D tensor of size " + nOutputPlane);

            if (outputWidth < 1 || outputHeight < 1)
                throw new InvalidOperationException(string.Format(
                    "Output size too small; calculated output size = ({0}x{1}x{2}", nOutputPlane, outputHeight, outputWidth));

            if (nInputPlane * cd.kW * cd.kH != weight.Shape[1])
                throw new InvalidOperationException(
                    string.Format("Input has incorrect number of channels. Got {0}, expected {1}", nInputPlane, weight.Shape[1] / ((float)(cd.kW * cd.kH))));

            if (input.DimensionCount != 4)
                throw new InvalidOperationException("4D input expected (NCHW order)");


            if (finput.Shape[0] != n || finput.Shape[1] != cd.kW * cd.kH * nInputPlane || finput.Shape[2] != outputHeight * outputWidth)
                throw new InvalidOperationException("finput is incorrect size");

            if (output.Shape[0] != n || output.Shape[1] != nOutputPlane || output.Shape[2] != outputHeight || output.Shape[3] != outputWidth)
                throw new InvalidOperationException("output is incorrect size");

            for (int i = 0; i < n; ++i)
            {
                using (var input_i = input.Select(0, i))
                using (var output_i = output.Select(0, i))
                using (var finput_i = finput.Select(0, i))
                {
                    Conv2ForwardFrame(input_i, output_i, weight, bias, finput_i,
                        cd.kW, cd.kH, cd.dW, cd.dW, cd.padW, cd.padH,
                        nInputPlane, inputWidth, inputHeight,
                        nOutputPlane, outputWidth, outputHeight);
                }
            }
        }

        /// <summary>
        /// Conv2s the forward frame.
        /// </summary>
        /// <param name="input">The input.</param>
        /// <param name="output">The output.</param>
        /// <param name="weight">The weight.</param>
        /// <param name="bias">The bias.</param>
        /// <param name="finput">The finput.</param>
        /// <param name="kW">The k w.</param>
        /// <param name="kH">The k h.</param>
        /// <param name="dW">The d w.</param>
        /// <param name="dH">The d h.</param>
        /// <param name="padW">The pad w.</param>
        /// <param name="padH">The pad h.</param>
        /// <param name="nInputPlane">The n input plane.</param>
        /// <param name="inputWidth">Width of the input.</param>
        /// <param name="inputHeight">Height of the input.</param>
        /// <param name="nOutputPlane">The n output plane.</param>
        /// <param name="outputWidth">Width of the output.</param>
        /// <param name="outputHeight">Height of the output.</param>
        private static void Conv2ForwardFrame(Tensor input, Tensor output, Tensor weight, Tensor bias, Tensor finput,
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH,
          long nInputPlane,
          long inputWidth,
          long inputHeight,
          long nOutputPlane,
          long outputWidth,
          long outputHeight)
        {
            var inputRef = NativeWrapper.AllocTensorRef(input);
            var finputRef = NativeWrapper.AllocTensorRef(finput);

            var inputPtr = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(TensorRef64)));
            Marshal.StructureToPtr(inputRef, inputPtr, false);
            var finputPtr = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(TensorRef64)));
            Marshal.StructureToPtr(finputRef, finputPtr, false);

            try
            {
                CpuOpsNative.TS_Unfolded_Copy(finputPtr, inputPtr, kW, kH, dW, dH, padW, padH, (int)nInputPlane, (int)inputWidth, (int)inputHeight, (int)outputWidth, (int)outputHeight);

                using (var output2d = output.View(nOutputPlane, outputHeight * outputWidth))
                {
                    if (bias != null)
                    {
                        using (var biasExp = bias.Expand(nOutputPlane, output2d.Shape[1]))
                        {
                            Ops.Copy(output2d, biasExp);
                        }
                    }
                    else
                    {
                        Ops.Fill(output, 0);
                    }

                    Ops.Addmm(output2d, 1, output2d, 1, weight, finput);
                }
            }
            finally
            {
                Marshal.FreeHGlobal(inputPtr);
                Marshal.FreeHGlobal(finputPtr);
                NativeWrapper.FreeTensorRef(inputRef);
                NativeWrapper.FreeTensorRef(finputRef);
            }
        }


        /// <summary>
        /// Conv2s the backward input.
        /// </summary>
        /// <param name="input">The input.</param>
        /// <param name="gradOutput">The grad output.</param>
        /// <param name="gradInput">The grad input.</param>
        /// <param name="weight">The weight.</param>
        /// <param name="finput">The finput.</param>
        /// <param name="fgradInput">The fgrad input.</param>
        /// <param name="cd">The cd.</param>
        /// <exception cref="InvalidOperationException">
        /// Number of output features must equal nOutputPlane
        /// or
        /// Kernel size should be greater than zero
        /// or
        /// stride should be greater than zero
        /// </exception>
        public static void Conv2BackwardInput(Tensor input, Tensor gradOutput, Tensor gradInput, Tensor weight, Tensor finput, Tensor fgradInput, ConvolutionDesc2d cd)
        {
            var nOutputPlane = weight.Shape[0];

            if (gradOutput.Shape[1] != nOutputPlane)
                throw new InvalidOperationException("Number of output features must equal nOutputPlane");

            if (cd.kW <= 0 && cd.kH <= 0)
                throw new InvalidOperationException("Kernel size should be greater than zero");

            if (cd.dW <= 0 && cd.dH <= 0)
                throw new InvalidOperationException("stride should be greater than zero");

            using (var weightT = weight.Transpose())
            {
                var n = input.Shape[0];

                for (int i = 0; i < n; ++i)
                {
                    using (var gradInput_i = gradInput.Select(0, i))
                    using (var gradOutput_i = gradOutput.Select(0, i))
                    using (var fgradInput_i = fgradInput.Select(0, i))
                    {
                        Conv2BackwardInputFrame(gradOutput_i, gradInput_i, weightT, fgradInput_i, cd);
                    }
                }
            }
        }

        /// <summary>
        /// Conv2s the backward input frame.
        /// </summary>
        /// <param name="gradOutput">The grad output.</param>
        /// <param name="gradInput">The grad input.</param>
        /// <param name="weight">The weight.</param>
        /// <param name="fgradInput">The fgrad input.</param>
        /// <param name="cd">The cd.</param>
        private static void Conv2BackwardInputFrame(Tensor gradOutput, Tensor gradInput, Tensor weight, Tensor fgradInput, ConvolutionDesc2d cd)
        {
            using (var gradOutput2d = gradOutput.View(gradOutput.Shape[0], gradOutput.Shape[1] * gradOutput.Shape[2]))
            {
                Ops.Addmm(fgradInput, 0, fgradInput, 1, weight, gradOutput2d);
            }

            Ops.Fill(gradInput, 0);

            IntPtr fgradInputPtr, gradInputPtr;
            using (NativeWrapper.BuildTensorRefPtr(fgradInput, out fgradInputPtr))
            using (NativeWrapper.BuildTensorRefPtr(gradInput, out gradInputPtr))
            {
                CpuOpsNative.TS_Unfolded_Acc(fgradInputPtr, gradInputPtr, cd.kW, cd.kH, cd.dW, cd.dH, cd.padW, cd.padH,
                (int)gradInput.Shape[0], (int)gradInput.Shape[2], (int)gradInput.Shape[1],
                (int)gradOutput.Shape[2], (int)gradOutput.Shape[1]);
            }
        }

        /// <summary>
        /// Conv2s the backward filter.
        /// </summary>
        /// <param name="input">The input.</param>
        /// <param name="gradOutput">The grad output.</param>
        /// <param name="gradWeight">The grad weight.</param>
        /// <param name="gradBias">The grad bias.</param>
        /// <param name="finput">The finput.</param>
        /// <param name="fgradInput">The fgrad input.</param>
        /// <param name="cd">The cd.</param>
        /// <exception cref="InvalidOperationException">
        /// Number of output features must equal nOutputPlane
        /// or
        /// Kernel size should be greater than zero
        /// or
        /// stride should be greater than zero
        /// </exception>
        public static void Conv2BackwardFilter(Tensor input, Tensor gradOutput, Tensor gradWeight, Tensor gradBias, Tensor finput, Tensor fgradInput, ConvolutionDesc2d cd)
        {
            var nOutputPlane = gradWeight.Shape[0];
            var n = input.Shape[0];

            if (gradOutput.Shape[1] != nOutputPlane)
                throw new InvalidOperationException("Number of output features must equal nOutputPlane");

            if (cd.kW <= 0 && cd.kH <= 0)
                throw new InvalidOperationException("Kernel size should be greater than zero");

            if (cd.dW <= 0 && cd.dH <= 0)
                throw new InvalidOperationException("stride should be greater than zero");

            for (int i = 0; i < n; ++i)
            {
                using (var gradOutput_i = gradOutput.Select(0, i))
                using (var finput_i = finput.Select(0, i))
                {
                    Conv2BackwardFilterFrame(gradOutput_i, gradWeight, gradBias, finput_i, cd);
                }
            }
        }

        /// <summary>
        /// Conv2s the backward filter frame.
        /// </summary>
        /// <param name="gradOutput">The grad output.</param>
        /// <param name="gradWeight">The grad weight.</param>
        /// <param name="gradBias">The grad bias.</param>
        /// <param name="finput">The finput.</param>
        /// <param name="cd">The cd.</param>
        private static void Conv2BackwardFilterFrame(Tensor gradOutput, Tensor gradWeight, Tensor gradBias, Tensor finput, ConvolutionDesc2d cd)
        {
            using (var gradOutput2d = gradOutput.View(gradOutput.Shape[0], gradOutput.Shape[1] * gradOutput.Shape[2]))
            using (var finputT = finput.Transpose())
            {
                Ops.Addmm(gradWeight, 1, gradWeight, 1, gradOutput2d, finputT);
                Ops.Sum(gradBias, gradOutput2d, 1);
            }
        }
    }
}
