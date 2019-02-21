// ***********************************************************************
// Assembly         : SiaNet.Backend.TensorSharp.CUDA91
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="SpatialConvolution.cs" company="SiaNet.Backend.TensorSharp.CUDA91">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SiaNet.Backend.TensorSharp.Cpu;
using SiaNet.Backend.TensorSharp.CUDA.DeviceCode;

namespace SiaNet.Backend.TensorSharp.CUDA
{
    /// <summary>
    /// Class SpatialConvolution.
    /// </summary>
    public class SpatialConvolution
    {
        /// <summary>
        /// The im2col kernels
        /// </summary>
        private readonly Im2ColCuda im2colKernels = new Im2ColCuda();

        /// <summary>
        /// Initializes a new instance of the <see cref="SpatialConvolution"/> class.
        /// </summary>
        public SpatialConvolution()
        {
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
            return new long[] { cd.kW * cd.kH * inputSizes[1], outputSizes[2] * outputSizes[3] };
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
        public void Conv2Forward(NDArray input, NDArray output, NDArray weight, NDArray bias, NDArray finput, ConvolutionDesc2d cd)
        {
            var batchSize = input.Shape[0];
            var nInputPlane = input.Shape[1];
            var inputWidth = input.Shape[3];
            var inputHeight = input.Shape[2];
            var nOutputPlane = weight.Shape[0];

            var outputWidth = (inputWidth + 2 * cd.padW - cd.kW) / cd.dW + 1;
            var outputHeight = (inputHeight + 2 * cd.padH - cd.kH) / cd.dH + 1;


            for (long i = 0; i < batchSize; ++i)
            {
                using (var input_i = input.Select(0, i))
                using (var output_i = output.Select(0, i))
                {
                    using (var output2d = output_i.View(nOutputPlane, outputHeight * outputWidth))
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
                            Ops.Fill(output_i, 0);
                        }

                        im2colKernels.Im2Col(input_i, finput, (int)nInputPlane, (int)inputHeight, (int)inputWidth,
                            cd.kH, cd.kW, cd.padH, cd.padW, cd.dH, cd.dW, 1, 1);

                        Ops.Addmm(output2d, 1, output2d, 1, weight, finput);
                    }

                }
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
        public void Conv2BackwardInput(NDArray input, NDArray gradOutput, NDArray gradInput, NDArray weight, NDArray finput, NDArray fgradInput, ConvolutionDesc2d cd)
        {
            var nOutputPlane = weight.Shape[0];
            var batchSize = input.Shape[0];

            var nInputPlane = input.Shape[1];
            var inputWidth = input.Shape[3];
            var inputHeight = input.Shape[2];

            var outputWidth = (inputWidth + 2 * cd.padW - cd.kW) / cd.dW + 1;
            var outputHeight = (inputHeight + 2 * cd.padH - cd.kH) / cd.dH + 1;


            for (long i = 0; i < batchSize; ++i)
            {
                using (var gradInput_i = gradInput.Select(0, i))
                using (var gradOutput_i = gradOutput.Select(0, i))
                using (var gradOutput_i2d = gradOutput_i.View(nOutputPlane, outputHeight * outputWidth))
                using (var weightT = weight.Transpose())
                {
                    Ops.Addmm(fgradInput, 0, fgradInput, 1, weightT, gradOutput_i2d);

                    im2colKernels.Col2Im(fgradInput, gradInput_i, (int)nInputPlane, (int)inputHeight, (int)inputWidth,
                        cd.kH, cd.kW, cd.padH, cd.padW, cd.dH, cd.dW, 1, 1);
                }
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
        public void Conv2BackwardFilter(NDArray input, NDArray gradOutput, NDArray gradWeight, NDArray gradBias, NDArray finput, NDArray fgradInput, ConvolutionDesc2d cd)
        {
            var nOutputPlane = gradWeight.Shape[0];
            var batchSize = input.Shape[0];

            var nInputPlane = input.Shape[1];
            var inputWidth = input.Shape[3];
            var inputHeight = input.Shape[2];

            var outputWidth = (inputWidth + 2 * cd.padW - cd.kW) / cd.dW + 1;
            var outputHeight = (inputHeight + 2 * cd.padH - cd.kH) / cd.dH + 1;

            for (long i = 0; i < batchSize; ++i)
            {
                using (var input_i = input.Select(0, i))
                using (var gradOutput_i = gradOutput.Select(0, i))
                {
                    im2colKernels.Im2Col(input_i, finput, (int)nInputPlane, (int)inputHeight, (int)inputWidth,
                        cd.kH, cd.kW, cd.padH, cd.padW, cd.dH, cd.dW, 1, 1);

                    using (var gradOutput2d = gradOutput_i.View(gradOutput_i.Shape[0], gradOutput_i.Shape[1] * gradOutput_i.Shape[2]))
                    using (var finputT = finput.Transpose())
                    {
                        Ops.Addmm(gradWeight, 1, gradWeight, 1, gradOutput2d, finputT);
                        Ops.Sum(gradBias, gradOutput2d, 1);
                    }

                }
            }
        }
    }
}
