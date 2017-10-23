using CNTK;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet
{
    public class Losses
    {
        public static Function Get(string loss, Variable labels, Variable predictions)
        {
            switch (loss.Trim().ToLower())
            {
                case OptLosses.BinaryCrossEntropy:
                    return BinaryCrossEntropy(labels, predictions);
                case OptLosses.CrossEntropy:
                    return CrossEntropy(labels, predictions);
                case OptLosses.KullbackLeiblerDivergence:
                    return KullbackLeiblerDivergence(labels, predictions);
                case OptLosses.MeanAbsoluteError:
                    return MeanAbsError(labels, predictions);
                case OptLosses.MeanAbsolutePercentageError:
                    return MeanAbsPercentageError(labels, predictions);
                case OptLosses.MeanSquaredError:
                    return MeanSquaredError(labels, predictions);
                case OptLosses.MeanSquaredLogError:
                    return MeanSquaredLogError(labels, predictions);
                case OptLosses.Poisson:
                    return Poisson(labels, predictions);
                case OptLosses.SparseCrossEntropy:
                    return SparseCrossEntropy(labels, predictions);
                default:
                    throw new NotImplementedException(string.Format("{0} is not implemented", loss));
            }
        }

        internal static Function MeanSquaredError(Variable labels, Variable predictions)
        {
            return CNTKLib.ReduceMean(CNTKLib.Square(CNTKLib.Minus(predictions, labels)), new Axis(-1));
        }

        internal static Function MeanAbsError(Variable labels, Variable predictions)
        {
            return CNTKLib.ReduceMean(CNTKLib.Abs(CNTKLib.Minus(predictions, labels)), new Axis(-1));
        }

        internal static Function MeanAbsPercentageError(Variable labels, Variable predictions)
        {
            var diff = CNTKLib.ElementDivide(CNTKLib.Abs(CNTKLib.Minus(predictions, labels)), CNTKLib.Clip(CNTKLib.Abs(labels), Utility.CreateParamVar(float.Epsilon), Utility.CreateParamVar(float.MaxValue)));
            var mean = CNTKLib.ReduceMean(diff, new Axis(-1));
            
            return CNTKLib.ElementTimes(Utility.CreateParamVar(100), mean);
        }

        internal static Function MeanSquaredLogError(Variable labels, Variable predictions)
        {
            var predLog = CNTKLib.Log(CNTKLib.Clip(predictions, Utility.CreateParamVar(float.Epsilon), Utility.CreateParamVar(float.MaxValue)));
            var labelsLog = CNTKLib.Log(CNTKLib.Clip(labels, Utility.CreateParamVar(float.Epsilon), Utility.CreateParamVar(float.MaxValue)));
            return CNTKLib.ReduceMean(CNTKLib.Square(CNTKLib.Minus(predLog, labelsLog)), new Axis(-1));
        }

        private static Function CrossEntropy(Variable labels, Variable predictions)
        {
            return CNTKLib.CrossEntropyWithSoftmax(predictions, labels);
        }

        private static Function SparseCrossEntropy(Variable labels, Variable predictions)
        {
            int newDim = labels.Shape.Dimensions.Aggregate((d1, d2) => d1 * d2);
            labels = CNTKLib.Reshape(labels, new int[] { newDim });
            return CNTKLib.CrossEntropyWithSoftmax(predictions, labels);
        }

        private static Function BinaryCrossEntropy(Variable labels, Variable predictions)
        {
            return CNTKLib.BinaryCrossEntropy(predictions, labels);
        }

        private static Function KullbackLeiblerDivergence(Variable labels, Variable predictions)
        {
            var label_t = CNTKLib.Clip(labels, Utility.CreateParamVar(float.Epsilon), Utility.CreateParamVar(1));
            var prediction_t = CNTKLib.Clip(predictions, Utility.CreateParamVar(float.Epsilon), Utility.CreateParamVar(1));
            return CNTKLib.ReduceSum(CNTKLib.ElementTimes(label_t, CNTKLib.Log(CNTKLib.ElementDivide(label_t, prediction_t))), new Axis(-1));
        }

        private static Function Poisson(Variable labels, Variable predictions)
        {
            return CNTKLib.ReduceMean(CNTKLib.Minus(predictions, CNTKLib.ElementTimes(labels, CNTKLib.Log(CNTKLib.Plus(predictions, Utility.CreateParamVar(float.Epsilon))))), new Axis(-1));
        }
    }
}
