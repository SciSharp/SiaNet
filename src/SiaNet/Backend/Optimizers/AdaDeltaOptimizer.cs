using System;
using System.Collections.Generic;

// ReSharper disable once CheckNamespace
namespace SiaNet.Backend
{

    public sealed class AdaDeltaOptimizer : BaseOptimizer
    {

        #region Fields

        private readonly Dictionary<int, NDArray> _AccG = new Dictionary<int, NDArray>();

        private readonly Dictionary<int, NDArray> _AccDelta = new Dictionary<int, NDArray>();

        #endregion

        #region Constructors

        public AdaDeltaOptimizer()
            : this(0)
        {
        }

        public AdaDeltaOptimizer(uint beginNumUpdate)
            : base(beginNumUpdate)
        {
            //this.SetParam("rho", 0.90f);
            //this.SetParam("epsilon", 1e-5);
        }

        #endregion

        #region Methods

        public override string GetOptimizerType()
        {
            return "adadelta";
        }

        public override void Update(int index, NDArray weight, NDArray grad)
        {
            if (weight == null)
                throw new ArgumentNullException(nameof(weight));
            if (grad == null)
                throw new ArgumentNullException(nameof(grad));

            if (!this._AccG.ContainsKey(index))
                this.CreateState(index, weight);

            var rho = float.Parse(this.Params["rho"]);
            var epsilon = float.Parse(this.Params["epsilon"]);
            var wd = this.GetWeightDecay(index);
            this.UpdateCount(index);

            if (this.Params.ContainsKey("rescale_grad"))
                grad *= float.Parse(this.Params["rescale_grad"]);

            if (this.Params.ContainsKey("clip_gradient"))
                Clip(ref grad, float.Parse(this.Params["clip_gradient"]));

            //auto & acc_g = *acc_g_[index];
            //auto & acc_delta = *acc_delta_[index];
            //acc_g *= rho;
            //acc_g += grad * grad * (1.0f - rho);

            //auto delta = _sqrt(acc_delta + epsilon) / _sqrt(acc_g + epsilon) * grad;
            //acc_delta *= rho;
            //acc_delta += delta * delta * (1.0f - rho);
            //weight *= 1.0f - wd;
            //weight -= delta;
            var accG = this._AccG[index];
            var accDelta = this._AccDelta[index];
            using (var tmp2 = grad * grad)
            using (var tmp3 = tmp2 * (1.0f - rho))
            {
                accG.Multiply(rho);
                accG.Add(tmp3);

                using (var tmp4 = accDelta + epsilon)
                using (var tmp5 = accG + epsilon)
                using (var tmp6 = Sqrt(tmp4))
                using (var tmp7 = Sqrt(tmp5))
                using (var tmp8 = tmp6 / tmp7)
                using (var delta = tmp8 * grad)
                using (var tmp11 = delta * delta)
                using (var tmp13 = tmp11 * (1.0f - wd))
                {
                    accDelta.Multiply(rho);
                    accDelta.Add(tmp13);

                    weight.Multiply(1.0f - wd);
                    weight.Subtract(delta);
                }
            }
        }

        #region Overrids

        protected override void CreateState(int index, NDArray weight)
        {
            this._AccG[index] = new NDArray(weight.GetShape());
            this._AccG[index].Set(0);
            this._AccDelta[index] = new NDArray(weight.GetShape());
            this._AccDelta[index].Set(0);
        }

        protected override void DisposeUnmanaged()
        {
            base.DisposeUnmanaged();

            foreach (var it in this._AccG)
                it.Value?.Dispose();

            foreach (var it in this._AccDelta)
                it.Value?.Dispose();
        }

        #endregion

        #endregion

    }

}
