namespace SiaNet
{
    using CNTK;
    using SiaNet.Common;
    using System;
    using System.Linq;

    /// <summary>
    /// A loss function (or objective function, or optimization score function) is one of the three parameters required to compile a model.The actual optimized objective is the mean of the output array across all datapoints.
    /// <see cref="OptLosses"/>
    /// </summary>
    internal class Losses
    {
        internal static Function Get(string loss, Variable labels, Variable predictions)
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
                case OptLosses.CTC:
                    return CTC(labels, predictions);
                default:
                    throw new NotImplementedException(string.Format("{0} is not implemented", loss));
            }
        }

        /// <summary>
        /// Means the squared error.
        /// </summary>
        /// <param name="labels">The labels.</param>
        /// <param name="predictions">The predictions.</param>
        /// <returns>Function.</returns>
        internal static Function MeanSquaredError(Variable labels, Variable predictions)
        {
            return CNTKLib.ReduceMean(CNTKLib.Square(CNTKLib.Minus(predictions, labels)), new Axis(-1));
        }

        /// <summary>
        /// Means the abs error.
        /// </summary>
        /// <param name="labels">The labels.</param>
        /// <param name="predictions">The predictions.</param>
        /// <returns>Function.</returns>
        internal static Function MeanAbsError(Variable labels, Variable predictions)
        {
            return CNTKLib.ReduceMean(CNTKLib.Abs(CNTKLib.Minus(predictions, labels)), new Axis(-1));
        }

        /// <summary>
        /// Means the abs percentage error.
        /// </summary>
        /// <param name="labels">The labels.</param>
        /// <param name="predictions">The predictions.</param>
        /// <returns>Function.</returns>
        internal static Function MeanAbsPercentageError(Variable labels, Variable predictions)
        {
            var diff = CNTKLib.ElementDivide(CNTKLib.Abs(CNTKLib.Minus(predictions, labels)), CNTKLib.Clip(CNTKLib.Abs(labels), Utility.CreateParamVar(float.Epsilon), Utility.CreateParamVar(float.MaxValue)));
            var mean = CNTKLib.ReduceMean(diff, new Axis(-1));
            
            return CNTKLib.ElementTimes(Utility.CreateParamVar(100), mean);
        }

        /// <summary>
        /// Means the squared log error.
        /// </summary>
        /// <param name="labels">The labels.</param>
        /// <param name="predictions">The predictions.</param>
        /// <returns>Function.</returns>
        internal static Function MeanSquaredLogError(Variable labels, Variable predictions)
        {
            var predLog = CNTKLib.Log(CNTKLib.Clip(predictions, Utility.CreateParamVar(float.Epsilon), Utility.CreateParamVar(float.MaxValue)));
            var labelsLog = CNTKLib.Log(CNTKLib.Clip(labels, Utility.CreateParamVar(float.Epsilon), Utility.CreateParamVar(float.MaxValue)));
            return CNTKLib.ReduceMean(CNTKLib.Square(CNTKLib.Minus(predLog, labelsLog)), new Axis(-1));
        }

        /// <summary>
        /// Crosses the entropy.
        /// </summary>
        /// <param name="labels">The labels.</param>
        /// <param name="predictions">The predictions.</param>
        /// <returns>Function.</returns>
        private static Function CrossEntropy(Variable labels, Variable predictions)
        {
            return CNTKLib.CrossEntropyWithSoftmax(predictions, labels);
        }

        /// <summary>
        /// Connectionist Temporal Classification is a loss function useful for performing supervised learning on sequence data, without needing an alignment between input data and labels. For example, CTC can be used to train end-to-end systems for speech recognition
        /// </summary>
        /// <param name="labels">The labels.</param>
        /// <param name="predictions">The predictions.</param>
        /// <returns></returns>
        private static Function CTC(Variable labels, Variable predictions)
        {
            return CNTKLib.EditDistanceError(predictions, labels, 0, 1, 1, true, new SizeTVector(1) { (uint)labels.Shape.TotalSize });
        }

        /// <summary>
        /// Sparses the cross entropy.
        /// </summary>
        /// <param name="labels">The labels.</param>
        /// <param name="predictions">The predictions.</param>
        /// <returns>Function.</returns>
        private static Function SparseCrossEntropy(Variable labels, Variable predictions)
        {
            int newDim = labels.Shape.Dimensions.Aggregate((d1, d2) => d1 * d2);
            labels = CNTKLib.Reshape(labels, new int[] { newDim });
            return CNTKLib.CrossEntropyWithSoftmax(predictions, labels);
        }

        /// <summary>
        /// Binaries the cross entropy.
        /// </summary>
        /// <param name="labels">The labels.</param>
        /// <param name="predictions">The predictions.</param>
        /// <returns>Function.</returns>
        private static Function BinaryCrossEntropy(Variable labels, Variable predictions)
        {
            return CNTKLib.BinaryCrossEntropy(predictions, labels);
        }

        /// <summary>
        /// Kullbacks the leibler divergence.
        /// </summary>
        /// <param name="labels">The labels.</param>
        /// <param name="predictions">The predictions.</param>
        /// <returns>Function.</returns>
        private static Function KullbackLeiblerDivergence(Variable labels, Variable predictions)
        {
            var label_t = CNTKLib.Clip(labels, Utility.CreateParamVar(float.Epsilon), Utility.CreateParamVar(1));
            var prediction_t = CNTKLib.Clip(predictions, Utility.CreateParamVar(float.Epsilon), Utility.CreateParamVar(1));
            return CNTKLib.ReduceSum(CNTKLib.ElementTimes(label_t, CNTKLib.Log(CNTKLib.ElementDivide(label_t, prediction_t))), new Axis(-1));
        }

        /// <summary>
        /// Poissons the specified labels.
        /// </summary>
        /// <param name="labels">The labels.</param>
        /// <param name="predictions">The predictions.</param>
        /// <returns>Function.</returns>
        private static Function Poisson(Variable labels, Variable predictions)
        {
            return CNTKLib.ReduceMean(CNTKLib.Minus(predictions, CNTKLib.ElementTimes(labels, CNTKLib.Log(CNTKLib.Plus(predictions, Utility.CreateParamVar(float.Epsilon))))), new Axis(-1));
        }
    }
}
