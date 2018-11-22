using System;
using System.Collections.Generic;
using System.Diagnostics;
using CNTK;
using SiaNet.Layers;

namespace SiaNet
{
    internal class LSTM2
    {
        private static Function Embedding(Variable input, int embeddingDim)
        {
            Debug.Assert(input.Shape.Rank == 1);
            var inputDim = input.Shape[0];
            var embeddingParameters = new Parameter(new[] {embeddingDim, inputDim}, DataType.Float,
                CNTKLib.GlorotUniformInitializer(), GlobalParameters.Device);

            return CNTKLib.Times(embeddingParameters, input);
        }

        private static Tuple<Function, Function> LSTMPCellWithSelfStabilization<ElementType>(
            Variable input,
            Variable prevOutput,
            Variable prevCellState)
        {
            var outputDim = prevOutput.Shape[0];
            var cellDim = prevCellState.Shape[0];

            var isFloatType = typeof(ElementType).Equals(typeof(float));
            var dataType = isFloatType ? DataType.Float : DataType.Double;

            Func<int, Parameter> createBiasParam;

            if (isFloatType)
            {
                createBiasParam = dim => new Parameter(new[] {dim}, 0.01f, GlobalParameters.Device, "");
            }
            else
            {
                createBiasParam = dim => new Parameter(new[] {dim}, 0.01, GlobalParameters.Device, "");
            }

            uint seed2 = 1;
            Func<int, Parameter> createProjectionParam = oDim => new Parameter(new[] {oDim, NDShape.InferredDimension},
                dataType, CNTKLib.GlorotUniformInitializer(1.0, 1, 0, seed2++), GlobalParameters.Device);

            Func<int, Parameter> createDiagWeightParam = dim =>
                new Parameter(new[] {dim}, dataType, CNTKLib.GlorotUniformInitializer(1.0, 1, 0, seed2++),
                    GlobalParameters.Device);

            var stabilizedPrevOutput = Stabilize<ElementType>(prevOutput, GlobalParameters.Device);
            var stabilizedPrevCellState = Stabilize<ElementType>(prevCellState, GlobalParameters.Device);

            Func<Variable> projectInput = () =>
                createBiasParam(cellDim) + createProjectionParam(cellDim) * input;

            // Input gate
            var it =
                CNTKLib.Sigmoid(
                    (Variable) (projectInput() + createProjectionParam(cellDim) * stabilizedPrevOutput) +
                    CNTKLib.ElementTimes(createDiagWeightParam(cellDim), stabilizedPrevCellState));
            var bit = CNTKLib.ElementTimes(
                it,
                CNTKLib.Tanh(projectInput() + createProjectionParam(cellDim) * stabilizedPrevOutput));

            // Forget-me-not gate
            var ft = CNTKLib.Sigmoid(
                (Variable) (
                    projectInput() + createProjectionParam(cellDim) * stabilizedPrevOutput) +
                CNTKLib.ElementTimes(createDiagWeightParam(cellDim), stabilizedPrevCellState));
            var bft = CNTKLib.ElementTimes(ft, prevCellState);

            var ct = (Variable) bft + bit;

            // Output gate
            var ot = CNTKLib.Sigmoid(
                (Variable) (projectInput() + createProjectionParam(cellDim) * stabilizedPrevOutput) +
                CNTKLib.ElementTimes(createDiagWeightParam(cellDim),
                    Stabilize<ElementType>(ct, GlobalParameters.Device)));
            var ht = CNTKLib.ElementTimes(ot, CNTKLib.Tanh(ct));

            var c = ct;
            var h = outputDim != cellDim
                ? createProjectionParam(outputDim) * Stabilize<ElementType>(ht, GlobalParameters.Device)
                : ht;

            return new Tuple<Function, Function>(h, c);
        }


        private static Tuple<Function, Function> LSTMPComponentWithSelfStabilization<ElementType>(
            Variable input,
            NDShape outputShape,
            NDShape cellShape,
            Func<Variable, Function> recurrenceHookH,
            Func<Variable, Function> recurrenceHookC)
        {
            var dh = Variable.PlaceholderVariable(outputShape, input.DynamicAxes);
            var dc = Variable.PlaceholderVariable(cellShape, input.DynamicAxes);

            var LSTMCell = LSTMPCellWithSelfStabilization<ElementType>(input, dh, dc);
            var actualDh = recurrenceHookH(LSTMCell.Item1);
            var actualDc = recurrenceHookC(LSTMCell.Item2);

            // Form the recurrence loop by replacing the dh and dc placeholders with the actualDh and actualDc
            LSTMCell.Item1.ReplacePlaceholders(new Dictionary<Variable, Variable> {{dh, actualDh}, {dc, actualDc}});

            return new Tuple<Function, Function>(LSTMCell.Item1, LSTMCell.Item2);
        }

        /// <summary>
        ///     Build a one direction recurrent neural network (RNN) with long-short-term-memory (LSTM) cells.
        ///     http://colah.github.io/posts/2015-08-Understanding-LSTMs/
        /// </summary>
        /// <param name="input">the input variable</param>
        /// <param name="numOutputClasses">number of output classes</param>
        /// <param name="embeddingDim">dimension of the embedding layer</param>
        /// <param name="LSTMDim">LSTM output dimension</param>
        /// <param name="cellDim">cell dimension</param>
        /// <param name="device">CPU or GPU device to run the model</param>
        /// <param name="outputName">name of the model output</param>
        /// <returns>the RNN model</returns>
        private static Function LSTMSequenceClassifierNet(
            Variable input,
            int numOutputClasses,
            int embeddingDim,
            int LSTMDim,
            int cellDim)
        {
            var embeddingFunction = new Embedding(embeddingDim).ToFunction(input);
            Func<Variable, Function> pastValueRecurrenceHook = x => CNTKLib.PastValue(x);
            var LSTMFunction = LSTMPComponentWithSelfStabilization<float>(
                embeddingFunction,
                new[] {LSTMDim},
                new[] {cellDim},
                pastValueRecurrenceHook,
                pastValueRecurrenceHook).Item1;
            var thoughtVectorFunction = CNTKLib.SequenceLast(LSTMFunction);

            return new Dense(numOutputClasses).ToFunction((Data.Function) thoughtVectorFunction);
        }

        private static Function Stabilize<TElementType>(Variable x, DeviceDescriptor device)
        {
            Constant f, fInv;

            if (typeof(TElementType) == typeof(float))
            {
                f = Constant.Scalar(4.0f, device);
                fInv = Constant.Scalar(f.DataType, 1.0 / 4.0f);
            }
            else
            {
                f = Constant.Scalar(4.0, device);
                fInv = Constant.Scalar(f.DataType, 1.0 / 4.0f);
            }

            var beta = CNTKLib.ElementTimes(
                fInv,
                CNTKLib.Log(
                    Constant.Scalar(f.DataType, 1.0) +
                    CNTKLib.Exp(CNTKLib.ElementTimes(f,
                        new Parameter(new NDShape(), f.DataType, 0.99537863 /* 1/f*ln (e^f-1) */, device)))));

            return CNTKLib.ElementTimes(beta, x);
        }
    }
}