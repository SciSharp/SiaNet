using System;
using System.Collections.Generic;
using System.Linq;
using SiaNet.Data;
using SiaNet.Metrics;
using SiaNet.Optimizers;
using SiaNet.Trainer.DQN;

namespace SiaNet.Trainer
{
    public class DQNTrainer
    {
        protected const double Accuracy = 0.001d;
        protected readonly CompiledModel TargetModel;

        public CompiledModel Model;

        public DQNTrainer(CompiledModel model, int memoryCapacity = 100000)
        {
            DiscountFactorGrowSteps = 100000;
            LearningRateDecaySteps = 500000;
            ReplayMemory = new FixedSizeList<Tuple<float[], int, float, float[]>>(memoryCapacity, false);
            TargetModel = model;
            Reset();
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
        protected FixedSizeList<Tuple<float[], int, float, float[]>> ReplayMemory { get; set; }

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

        public virtual double Fit(
            IEnvironment trainEnvironment,
            int batchSize,
            OptimizerBase optimizer,
            MetricFunction lossMetric,
            Func<double, bool> ender,
            MetricFunction evaluationMetric = null,
            IEnvironment validationEnvironment = null,
            bool shuffle = false)
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
                    RandomAgentReplay(batchSize, optimizer, lossMetric, shuffle);

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
                    AgentReplay(batchSize, optimizer, lossMetric, shuffle);

                    state = stateNext;
                    totalReward += state.Reward;
                }

                rewards.Add(totalReward);
            }

            return rewards.Max();
        }

        public virtual void Reset()
        {
            Model = TargetModel.Clone();
            Steps = 0;
            LearningRate = InitialLearningRate;
            DiscountFactor = InitialDiscountFactor;
            ReplayMemory.Clear();
        }

        protected virtual int AgentAct(float[] state)
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


        protected virtual void AgentObserve(Tuple<float[], int, float, float[]> sample)
        {
            ReplayMemory.Add(sample);

            if (Steps > 0 && Steps % TargetUpdateFrequency == 0)
            {
                UpdateTargetModel();
            }

            Steps++;

            if (Steps % 23 == 0)
            {
                Console.SetCursorPosition(0, Console.CursorTop);
                Console.Write(Steps);
            }

            LearningRate = FinalLearningRate +
                           (InitialLearningRate - FinalLearningRate) * Math.Exp(-LearningRateDecay * Steps);
            DiscountFactor = FinalDiscountFactor +
                             (InitialDiscountFactor - FinalDiscountFactor) * Math.Exp(-DiscountFactorGrow * Steps);
        }

        protected virtual void AgentReplay(int batchSize, OptimizerBase optimizer, MetricFunction lossMetric, bool shuffle)
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
                    t[sample.Item2] = (float) (sample.Item3 + DiscountFactor * predictionTarget[i].Max());
                }

                data.AddFrame(sample.Item1, t);
            }

            Model.Fit(data, 1, batch.Length, optimizer, lossMetric);
        }

        protected virtual int RandomAgentAct(float[] state)
        {
            // random action
            return RandomGenerator.RandomInt(0, ActionShape.TotalSize);
        }

        protected virtual void RandomAgentObserve(Tuple<float[], int, float, float[]> sample)
        {
            ReplayMemory.Add(sample);
        }

        protected virtual void RandomAgentReplay(int batchSize, OptimizerBase optimizer, MetricFunction lossMetric, bool shuffle)
        {
        }

        protected virtual void UpdateTargetModel()
        {
            TargetModel.SetParameters(Model.GetParameters());
        }
    }
}