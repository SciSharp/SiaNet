using CNTK;
using SiaNet.Common;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet.NN
{
    public class Recurrent
    {
        public static Function LSTM(Variable layer, int dim, int? cellDim=null, string activation = OptActivations.Tanh, string recurrentActivation = OptActivations.Sigmoid, string weightInitializer = OptInitializers.GlorotUniform, string recurrentInitializer = OptInitializers.GlorotUniform, bool useBias = true, string biasInitializer = OptInitializers.Zeros, bool returnSequence = false)
        {
            cellDim = cellDim.HasValue ? cellDim : dim;
            Variable prevOutput = Variable.PlaceholderVariable(new int[] { dim }, layer.DynamicAxes);
            Variable prevCellState = cellDim.HasValue ? Variable.PlaceholderVariable(new int[] { cellDim.Value }, layer.DynamicAxes) : null;

            Func<int, Parameter> createBiasParam = (d) => new Parameter(new int[] { d }, DataType.Float, Initializers.Get(biasInitializer), GlobalParameters.Device);
            
            Func<int, Parameter> createProjectionParam = (oDim) => new Parameter(new int[] { oDim, NDShape.InferredDimension },
                    DataType.Float, Initializers.Get(weightInitializer), GlobalParameters.Device);

            Func<int, Parameter> createDiagWeightParam = (d) =>
                new Parameter(new int[] { d }, DataType.Float, Initializers.Get(recurrentInitializer), GlobalParameters.Device);

            Function stabilizedPrevOutput = Stabilize<float>(prevOutput, GlobalParameters.Device);
            Function stabilizedPrevCellState = prevCellState !=null ? Stabilize<float>(prevCellState, GlobalParameters.Device) : null;

            Func<Variable> projectInput = null;

            if (cellDim.HasValue)
                projectInput = () => createBiasParam(cellDim.Value) + (createProjectionParam(cellDim.Value) * layer);
            else
                projectInput = () => layer;

            //Input gate
            Function it = null;
            if(cellDim.HasValue)
            {
                it = Basic.Activation((Variable)(projectInput() + (createProjectionParam(cellDim.Value) * stabilizedPrevOutput)) + CNTKLib.ElementTimes(createDiagWeightParam(cellDim.Value), stabilizedPrevCellState), recurrentActivation);
            }
            else
            {
                it = Basic.Activation((Variable)(projectInput()), recurrentActivation);
            }

            Function bit = null;
            if(cellDim.HasValue)
            {
                bit = CNTKLib.ElementTimes(it, Basic.Activation(projectInput() + (createProjectionParam(cellDim.Value) * stabilizedPrevOutput), activation));
            }
            else
            {
                bit = CNTKLib.ElementTimes(it, Basic.Activation(projectInput(), activation));
            }

            // Forget-me-not gate
            Function ft = null;
            if(cellDim.HasValue)
            {
                ft = Basic.Activation((Variable)(projectInput() + (createProjectionParam(cellDim.Value) * stabilizedPrevOutput)) + CNTKLib.ElementTimes(createDiagWeightParam(cellDim.Value), stabilizedPrevCellState), recurrentActivation);
            }
            else
            {
                ft = Basic.Activation(projectInput(), recurrentActivation);
            }

            Function bft = prevCellState !=null ? CNTKLib.ElementTimes(ft, prevCellState) : ft;

            Function ct = (Variable)bft + bit;

            //Output gate
            Function ot = null;
            if(cellDim.HasValue)
            {
                ot = Basic.Activation((Variable)(projectInput() + (createProjectionParam(cellDim.Value) * stabilizedPrevOutput)) + CNTKLib.ElementTimes(createDiagWeightParam(cellDim.Value), Stabilize<float>(ct, GlobalParameters.Device)), recurrentActivation);
            }
            else
            {
                ot = Basic.Activation((Variable)(projectInput()) + Stabilize<float>(ct, GlobalParameters.Device), recurrentActivation);
            }

            Function ht = CNTKLib.ElementTimes(ot, CNTKLib.Tanh(ct));
            Function c = ct;
            Function h = (dim != cellDim) ? (createProjectionParam(dim) * Stabilize<float>(ht, GlobalParameters.Device)) : ht;

            Func<Variable, Function> recurrenceHookH = (x) => CNTKLib.PastValue(x);
            Func<Variable, Function> recurrenceHookC = (x) => CNTKLib.PastValue(x);

            var actualDh = recurrenceHookH(h);
            var actualDc = recurrenceHookC(c);

            if (prevCellState != null)
                h.ReplacePlaceholders(new Dictionary<Variable, Variable> { { prevOutput, actualDh }, { prevCellState, actualDc } });
            else
                h.ReplacePlaceholders(new Dictionary<Variable, Variable> { { prevOutput, actualDh } });

            if (returnSequence)
                return h;
            
            return CNTKLib.SequenceLast(h);
        }

        public static Function LSTM(int shape, int dim, int? cellDim = null, string activation = OptActivations.Tanh, string recurrentActivation = OptActivations.Sigmoid, string weightInitializer = OptInitializers.GlorotUniform, string recurrentInitializer = OptInitializers.GlorotUniform, bool useBias = true, string biasInitializer = OptInitializers.Zeros)
        {
            return LSTM(Variable.InputVariable(new int[] { shape }, DataType.Float, isSparse: true), dim, cellDim, activation, recurrentActivation, weightInitializer, recurrentInitializer, useBias, biasInitializer);
        }

        private static Function Stabilize<ElementType>(Variable x, DeviceDescriptor device)
        {
            bool isFloatType = typeof(ElementType).Equals(typeof(float));
            Constant f, fInv;
            if (isFloatType)
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
                    CNTKLib.Exp(CNTKLib.ElementTimes(f, new Parameter(new NDShape(), f.DataType, 0.99537863 /* 1/f*ln (e^f-1) */, device)))));
            return CNTKLib.ElementTimes(beta, x);
        }
    }
}
