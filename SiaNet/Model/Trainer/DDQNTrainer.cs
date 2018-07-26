using SiaNet.Model.Data;
using SiaNet.Model.Metrics;
using SiaNet.Model.Optimizers;

namespace SiaNet.Model.Trainer
{
    public class DDQNTrainer : DQNTrainer
    {
        public DDQNTrainer(CompiledModel model, int memoryCapacity = 100000) : base(model, memoryCapacity)
        {
        }

        protected override void AgentReplay(
            int batchSize,
            OptimizerBase optimizer,
            MetricFunction lossMetric,
            bool shuffle)
        {
            var batch = shuffle ? ReplayMemory.ToShuffledBatch(batchSize) : ReplayMemory.ToRandomBatch(batchSize);

            var states = new DataFrame(StateShape);
            var statesTarget = new DataFrame(StateShape);

            foreach (var sample in batch)
            {
                states.Add(sample.Item1);
                statesTarget.Add(sample.Item4 ?? new float[StateShape.TotalSize]);
            }

            var prediction = Model.Predict(states);
            var predictionOfTargetStates = Model.Predict(statesTarget);
            var predictionTarget = TargetModel.Predict(statesTarget);

            var data = new DataFrameList(StateShape, ActionShape);

            for (var i = 0; i < batch.Length; i++)
            {
                var sample = batch[i];

                var t = prediction[i];

                if (sample.Item4 == null)
                {
                    t[sample.Item2] = sample.Item3;
                }
                else
                {
                    var lastValue = float.MinValue;
                    var valueIndex = 0;

                    for (var j = 0; j < predictionOfTargetStates[i].Length; j++)
                    {
                        if (predictionOfTargetStates[i][j] > lastValue)
                        {
                            lastValue = predictionOfTargetStates[i][j];
                            valueIndex = j;
                        }
                    }

                    t[sample.Item2] = (float) (sample.Item3 + DiscountFactor * predictionTarget[i][valueIndex]);
                }

                data.AddFrame(sample.Item1, t);
            }

            Model.Fit(data, 1, batch.Length, optimizer, lossMetric);
        }
    }
}