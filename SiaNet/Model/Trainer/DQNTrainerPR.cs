using System;
using System.Collections.Generic;
using System.Linq;
using SiaNet.Model.Data;
using SiaNet.Model.Metrics;
using SiaNet.Model.Optimizers;
using SiaNet.Model.Trainer.DQN;

namespace SiaNet.Model.Trainer
{
    public class DQNTrainerPR
    {
        protected const double Accuracy = 0.001d;
        protected const double E = 0.01;
        protected const double A = 0.6;

        protected readonly CompiledModel TargetModel;

        public CompiledModel Model;

        public DQNTrainerPR(CompiledModel model, int memoryCapacity = 50000)
        {
            DiscountFactorGrowSteps = 100000;
            LearningRateDecaySteps = 500000;
            ReplayMemory = new SumTreeList<Tuple<float[], int, float, float[]>>(memoryCapacity);
            TargetModel = model;
            Reset();
        }

        protected int GetPriority(double error)
        {
            return (int)Math.Pow(error + E, A);
        }

        /**
         * Gets the shape of the expected actions matrix
         */
        public Shape ActionShape
        {
            get => Model.OutputShape;
        }

        /**
         * Gets the discount factor (gamma), 0 = Myopic (short-sighted), 1 = Hyperopic (far-sighted)
         */
        public double DiscountFactor { get; protected set; }

        /**
         * Gets or sets the speed of grow of the discount factor (gamma)
         */
        public double DiscountFactorGrow { get; set; }

        /**
         * Gets or sets the number of steps for discount factor to reach the final value, directly influencing the discount factor grow
         */
        public int DiscountFactorGrowSteps
        {
            get => (int) (-Math.Log(Accuracy) / DiscountFactorGrow);
            set => DiscountFactorGrow = -Math.Log(Accuracy) / value;
        }

        /**
         * Gets or sets the final discount factor (gamma)
         */
        public double FinalDiscountFactor { get; set; } = 0.99f;

        /**
         * Gets or sets the minimum or the final exploration rate
         */
        public double FinalLearningRate { get; set; } = 0.01d;

        /**
         * Gets or sets the initial discount factor (gamma)
         */
        public double InitialDiscountFactor { get; set; } = 0.2f;

        /**
         * Gets or sets the maximum or the initial exploration rate
         */
        public double InitialLearningRate { get; set; } = 1d;

        /**
         * Gets the current learning rate
         */
        public double LearningRate { get; protected set; }

        /**
         * Gets or sets the speed of decay of the exploration rate
         */
        public double LearningRateDecay { get; set; }

        /**
         * Gets or sets the number of steps for learning rate to reach the final value, directly influencing the speed of decay
         */
        public int LearningRateDecaySteps
        {
            get => (int) (-Math.Log(Accuracy) / LearningRateDecay);
            set => LearningRateDecay = -Math.Log(Accuracy) / value;
        }

        /**
         * Gets the capacity of the replay memory
         */
        public int MemoryCapacity
        {
            get => ReplayMemory.Capacity;
        }

        /**
         * Gets or sets a boolean value indicating if the memory should be filled with random interactions with the environment before starting the learning process
         */
        public bool RandomActionPreparation { get; set; } = true;

        /**
         * Gets or sets the reply memory
         */
        protected SumTreeList<Tuple<float[], int, float, float[]>> ReplayMemory { get; set; }

        /**
         * Gets the shape of the states matrix
         */
        public Shape StateShape
        {
            get => Model.InputShape;
        }

        /**
         * Gets the current learning step
         */
        public int Steps { get; protected set; }

        /**
         * Gets or sets the frequency in which the target function should get updated, set to 1 to disable function approximation altogether
         */
        public int TargetUpdateFrequency { get; set; } = 1000;

        public double Fit(
            IEnvironment trainEnvironment,
            int batchSize,
            OptimizerBase optimizer,
            MetricFunction lossMetric,
            Func<double, bool> ender,
            MetricFunction evaluationMetric = null,
            IEnvironment validationEnvironment = null)
        {
            var rewards = new List<double>();

            while (RandomActionPreparation && !ReplayMemory.IsFull)
            {
                var state = trainEnvironment.Reset();
                var totalReward = 0d;

                while (!state.IsEnded)
                {
                    var a = RandomAgentAct(state.State);
                    var stateNext = trainEnvironment.Step(a);

                    RandomAgentObserve(
                        new Tuple<float[], int, float, float[]>(state.State, a, stateNext.Reward, stateNext.State));

                    state = stateNext;
                    totalReward += state.Reward;
                }

                rewards.Add(totalReward);
            }

            while (!ender.Invoke(rewards.DefaultIfEmpty().Last()))
            {
                var state = trainEnvironment.Reset();
                var totalReward = 0d;

                while (!state.IsEnded)
                {
                    var a = AgentAct(state.State);
                    var stateNext = trainEnvironment.Step(a);

                    AgentObserve(
                        new Tuple<float[], int, float, float[]>(state.State, a, stateNext.Reward, stateNext.State));
                    AgentReplay(batchSize, optimizer, lossMetric);

                    state = stateNext;
                    totalReward += state.Reward;
                }

                rewards.Add(totalReward);
            }

            return rewards.Max();
        }

        public void Reset()
        {
            Model = TargetModel.Clone();
            Steps = 0;
            LearningRate = InitialLearningRate;
            DiscountFactor = InitialDiscountFactor;
            ReplayMemory.Clear();
        }

        private int AgentAct(float[] state)
        {
            if (RandomGenerator.RandomDouble(0, 1) < LearningRate)
            {
                // random action
                return RandomGenerator.RandomInt(0, ActionShape.TotalSize);
            }

            // from prediction
            var predict = Model.Predict(state);
            var maxF = float.MinValue;
            var maxFIndex = -1;

            for (var i = 0; i < predict.Length; i++)
            {
                if (predict[i] > maxF)
                {
                    maxF = predict[i];
                    maxFIndex = i;
                }
            }

            return maxFIndex;
        }


        private void AgentObserve(Tuple<float[], int, float, float[]> sample)
        {
            ReplayMemory.Add(GetPriority(AgentGetTargets(new[] { sample }).Item2[0]), sample);

            if (Steps > 0 && Steps % TargetUpdateFrequency == 0)
            {
                UpdateTargetModel();
            }

            Steps++;
            LearningRate = FinalLearningRate +
                           (InitialLearningRate - FinalLearningRate) * Math.Exp(-LearningRateDecay * Steps);
            DiscountFactor = FinalDiscountFactor +
                             (InitialDiscountFactor - FinalDiscountFactor) * Math.Exp(-DiscountFactorGrow * Steps);
        }

        protected Tuple<DataFrameList, float[]> AgentGetTargets(Tuple<float[], int, float, float[]>[] samples)
        {
            var states = new DataFrame(StateShape);
            var statesTarget = new DataFrame(StateShape);

            foreach (var sample in samples)
            {
                states.Add(sample.Item1);
                statesTarget.Add(sample.Item4 ?? new float[StateShape.TotalSize]);
            }

            var prediction = Model.Predict(states);
            var predictionOfTargetStates = Model.Predict(statesTarget);
            var predictionTarget = TargetModel.Predict(statesTarget);

            var data = new DataFrameList(StateShape, ActionShape);
            var errors = new float[samples.Length];

            for (var i = 0; i < samples.Length; i++)
            {
                var sample = samples[i];

                var t = prediction[i];
                var oldVal = t[sample.Item2];
                if (sample.Item4 == null)
                {
                    t[sample.Item2] = sample.Item3;
                }
                else
                {
                    float lastValue = float.MinValue;
                    int valueIndex = 0;
                    for (int j = 0; j < predictionOfTargetStates[i].Length; j++)
                    {
                        if (predictionOfTargetStates[i][j] > lastValue)
                        {
                            lastValue = predictionOfTargetStates[i][j];
                            valueIndex = j;
                        }
                    }
                    t[sample.Item2] = (float)(sample.Item3 + DiscountFactor * predictionTarget[i][valueIndex]);
                }

                data.AddFrame(sample.Item1, t);
                errors[i] = oldVal;
            }

            return new Tuple<DataFrameList, float[]>(data, errors);
        }

        private void AgentReplay(int batchSize, OptimizerBase optimizer, MetricFunction lossMetric)
        {
            var batch = ReplayMemory.ToBatch(batchSize);

            var data = AgentGetTargets(batch.Select(t => t.Item3).ToArray());

            var indexes = batch.Select(t => t.Item1).ToArray();
            for (int i = 0; i < batch.Count; i++)
            {
                ReplayMemory.Update(indexes[i], GetPriority(data.Item2[i]));
            }

            Model.Fit(data.Item1, 1, batch.Count, optimizer, lossMetric);
        }

        private int RandomAgentAct(float[] state)
        {
            // random action
            return RandomGenerator.RandomInt(0, ActionShape.TotalSize);
        }

        private void RandomAgentObserve(Tuple<float[], int, float, float[]> sample)
        {
            var error = Math.Abs(sample.Item3);
            ReplayMemory.Add(GetPriority(error) ,sample);
        }
        
        private void UpdateTargetModel()
        {
            TargetModel.SetParameters(Model.GetParameters());
        }
    }
}