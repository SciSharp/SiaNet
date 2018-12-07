using System;
using System.Collections.Generic;

// ReSharper disable once CheckNamespace
namespace SiaNet.Backend
{

    public sealed class AdaGradOptimizer : BaseOptimizer
    {

        #region Fields

        private readonly Dictionary<int, NDArray> _History = new Dictionary<int, NDArray>();

        #endregion

        #region Constructors

        public AdaGradOptimizer()
            : this(0)
        {
        }

        public AdaGradOptimizer(uint beginNumUpdate)
            : base(beginNumUpdate)
        {
            //this.SetParam("eps", 1e-7);
        }

        #endregion

        #region Methods

        public override string GetOptimizerType()
        {
            return "adagrad";
        }

        public override void Update(int index, NDArray weight, NDArray grad)
        {
            if (weight == null)
                throw new ArgumentNullException(nameof(weight));
            if (grad == null)
                throw new ArgumentNullException(nameof(grad));

            if (!this._History.ContainsKey(index))
                this.CreateState(index, weight);

            var eps = float.Parse(this.Params["eps"]);
            var lr = this.GetLearningRate(index);
            var wd = this.GetWeightDecay(index);
            this.UpdateCount(index);
            if (this.Params.ContainsKey("rescale_grad"))
                grad *= float.Parse(this.Params["rescale_grad"]);

            if (this.Params.ContainsKey("clip_gradient"))
                Clip(ref grad, float.Parse(this.Params["clip_gradient"]));

            //auto & history = *history_[index];
            //history += grad * grad;
            //weight -= (grad / _sqrt(history + eps) + weight * wd) * lr;
            var history = this._History[index];
            using (var tmp1 = grad * grad)
            {
                history.Add(tmp1);

                using (var tmp2 = history + eps)
                using (var tmp3 = Sqrt(tmp2))
                using (var tmp4 = weight * wd)
                using (var tmp5 = grad / tmp3)
                using (var tmp6 = tmp5 + tmp4)
                using (var tmp7 = tmp6 * lr)
                    weight.Subtract(tmp7);
            }
        }

        #region Overrids

        protected override void CreateState(int index, NDArray weight)
        {
            this._History[index] = new NDArray(weight.GetShape());
            this._History[index].Set(0);
        }

        protected override void DisposeUnmanaged()
        {
            base.DisposeUnmanaged();

            foreach (var it in this._History)
                it.Value?.Dispose();
        }

        #endregion

        #endregion

    }

}
