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
        public static Function LSTMCell(Variable layer, int dim, int? cellDim=null, string activation = OptActivations.Tanh, string recurrentActivation = OptActivations.Sigmoid, string weightInitializer = OptInitializers.GlorotUniform, string recurrentInitializer = OptInitializers.GlorotUniform, bool useBias = true, string biasInitializer = OptInitializers.Zeros)
        {
            Variable prevOutput = Variable.InputVariable(new int[] { dim }, DataType.Float, dynamicAxes: layer.DynamicAxes);
            Variable prevCellState = cellDim.HasValue ? Variable.InputVariable(new int[] { cellDim.Value }, DataType.Float, dynamicAxes: layer.DynamicAxes) : null;

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
                it = Basic.Activation((Variable)(projectInput() + stabilizedPrevOutput), recurrentActivation);
            }

            Function bit = null;
            if(cellDim.HasValue)
            {
                bit = CNTKLib.ElementTimes(it, Basic.Activation(projectInput() + (createProjectionParam(cellDim.Value) * stabilizedPrevOutput), activation));
            }
            else
            {
                bit = CNTKLib.ElementTimes(it, Basic.Activation(projectInput() + stabilizedPrevOutput, activation));
            }

            // Forget-me-not gate
            Function ft = null;
            if(cellDim.HasValue)
            {
                ft = Basic.Activation((Variable)(projectInput() + (createProjectionParam(cellDim.Value) * stabilizedPrevOutput)) + CNTKLib.ElementTimes(createDiagWeightParam(cellDim.Value), stabilizedPrevCellState), recurrentActivation);
            }
            else
            {
                ft = Basic.Activation(projectInput() + stabilizedPrevOutput, recurrentActivation);
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
                ot = Basic.Activation((Variable)(projectInput() + stabilizedPrevOutput) + Stabilize<float>(ct, GlobalParameters.Device), recurrentActivation);
            }

            Function ht = CNTKLib.ElementTimes(ot, CNTKLib.Tanh(ct));
            Function c = ct;
            Function h = (dim != cellDim) ? (createProjectionParam(dim) * Stabilize<float>(ht, GlobalParameters.Device)) : ht;

            Func<Variable, Function> recurrenceHookH = (x) => CNTKLib.PastValue(x);
            Func<Variable, Function> recurrenceHookC = (x) => CNTKLib.PastValue(x);

            Tuple<Function, Function> LSTMCell = new Tuple<Function, Function>(h, c);
            var actualDh = recurrenceHookH(h);
            var actualDc = recurrenceHookC(c);

            if (prevCellState != null)
                h.ReplacePlaceholders(new Dictionary<Variable, Variable> { { prevOutput, actualDh }, { prevCellState, actualDc } });
            else
                h.ReplacePlaceholders(new Dictionary<Variable, Variable> { { prevOutput, actualDh } });

            return CNTKLib.SequenceLast(h);
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
