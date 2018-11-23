using CNTK;

namespace SiaNet.Metrics
{
    public class ConnectionistTemporalClassification : MetricFunction
    {
        public ConnectionistTemporalClassification() : base(ConnectionistTemporalClassificationFunction)
        {
        }

        /// <summary>
        /// Connectionist Temporal Classification is a loss function useful for performing supervised learning on sequence data, without needing an alignment between input data and labels. For example, CTC can be used to train end-to-end systems for speech recognition
        /// </summary>
        protected static Data.Function ConnectionistTemporalClassificationFunction(Data.Variable labels, Data.Variable predictions)
        {
            return CNTKLib.EditDistanceError(predictions, labels, 0, 1, 1, true,
                new SizeTVector(1) { (uint)labels.Shape.TotalSize });
        }
    }
}