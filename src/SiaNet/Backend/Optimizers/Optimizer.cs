using System;
using System.Collections.Generic;
using System.Linq;

// ReSharper disable once CheckNamespace
namespace SiaNet.Backend
{

    public abstract class BaseOptimizer : DisposableMXNetObject
    {

        #region Fields

        protected static readonly OpMap OpMap;

        #endregion

        #region Constructors

        static BaseOptimizer()
        {
            OpMap = new OpMap();
        }

        protected BaseOptimizer(uint beginNumUpdate)
        {
            this.BeginNumUpdate = beginNumUpdate;
            this.NumUpdate = beginNumUpdate;

            this.Count = new Dictionary<int, uint>();
            this.Params = new Dictionary<string, string>();
            this.Params["lr"] = "0.01f";
            this.Params["wd"] = "0.f";
        }

        #endregion

        #region Properties

        protected uint BeginNumUpdate
        {
            get;
        }

        protected Dictionary<int, uint> Count
        {
            get;
        }

        private UniquePtr<LRScheduler> _LearningRateScheduler;

        protected UniquePtr<LRScheduler> LearningRateScheduler => this._LearningRateScheduler;

        protected uint NumUpdate
        {
            get;
            private set;
        }

        protected Dictionary<string, string> Params
        {
            get;
        }

        #endregion

        #region Methods

        protected static void Clip(ref NDArray data, float limit)
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));

            using (var op = new Operator("clip"))
            {
                op
                    .SetParam("a_min", -limit)
                    .SetParam("a_max", limit)
                    .SetInput("data", data)
                    .Invoke(data);
            }
        }

        protected virtual void CreateState(int index, NDArray weight)
        {
        }

        protected float GetLearningRate(int index)
        {
            if (null != this.LearningRateScheduler)
                return this.LearningRateScheduler.Ptr.GetLearningRate(this.NumUpdate);

            return float.Parse(this.Params["lr"].Replace("f", ""));
        }

        protected float GetWeightDecay(int index)
        {
            var wd = float.Parse(this.Params["wd"].Replace("f", ""));
            return wd;
        }

        public abstract string GetOptimizerType();

        protected string[] GetParamKeys_()
        {
            return this.Params.Select(iter => iter.Key).ToArray();
        }

        protected string[] GetParamValues_()
        {
            return this.Params.Select(iter => iter.Value).ToArray();
        }

        public BaseOptimizer SetLearningRateScheduler(UniquePtr<LRScheduler> lrScheduler)
        {
            Logging.CHECK(lrScheduler);
            UniquePtr<LRScheduler>.Move(lrScheduler, out this._LearningRateScheduler);
            this.LearningRateScheduler.Ptr.SetLearningRate(float.Parse(this.Params["lr"]));
            return this;
        }

        public BaseOptimizer SetParam(string name, object value)
        {
            this.Params[name] = value.ToString();
            return this;
        }

        protected static NDArray Sqrt(NDArray data)
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));

            using (var op = new Operator("sqrt"))
                return op.SetInput("data", data).Invoke()[0];
        }

        public abstract void Update(int index, NDArray weight, NDArray grad);

        protected uint UpdateCount(int index)
        {
            if (!this.Count.TryGetValue(index, out var value))
            {
                this.Count.Add(index, this.BeginNumUpdate);
                value = this.BeginNumUpdate;
            }

            var newCount = ++value;
            this.Count[index] = newCount;
            this.NumUpdate = Math.Max(this.NumUpdate, newCount);
            return newCount;
        }

        #endregion

    }
    
}
