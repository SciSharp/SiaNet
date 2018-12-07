using System;
using System.Collections.Generic;
using System.Globalization;
using SiaNet.Backend.Interop;
using NDArrayHandle = System.IntPtr;
using AtomicSymbolCreator = System.IntPtr;

// ReSharper disable once CheckNamespace
namespace SiaNet.Backend
{

    public sealed class AdamOptimizer : BaseOptimizer
    {

        #region Fields

        private readonly Dictionary<int, NDArray> _Mean = new Dictionary<int, NDArray>();

        private readonly Dictionary<int, NDArray> _Var = new Dictionary<int, NDArray>();

        private readonly AtomicSymbolCreator _UpdateHandle;

        #endregion

        #region Constructors

        public AdamOptimizer()
            : this(0)
        {

        }

        public AdamOptimizer(uint beginNumUpdate)
            : base(beginNumUpdate)
        {
            this._UpdateHandle = OpMap.GetSymbolCreator("adam_update");
            this.SetParam("beta1", 0.9f);
            this.SetParam("beta2", 0.999f);
            this.SetParam("epsilon", 1e-8);
        }

        #endregion

        #region Methods

        public override string GetOptimizerType()
        {
            return "adam";
        }

        public override void Update(int index, NDArray weight, NDArray grad)
        {
            if (weight == null)
                throw new ArgumentNullException(nameof(weight));
            if (grad == null)
                throw new ArgumentNullException(nameof(grad));

            if (!this._Mean.ContainsKey(index))
                this.CreateState(index, weight);

            this.Params["lr"] = this.GetLearningRate(index).ToString(CultureInfo.InvariantCulture);
            this.Params["wd"] = this.GetWeightDecay(index).ToString(CultureInfo.InvariantCulture);
            this.UpdateCount(index);
            var keys = this.GetParamKeys_();
            var values = this.GetParamValues_();
            Logging.CHECK_EQ(keys.Length, values.Length);

            //var lr = double.Parse(params_["lr"]);
            //var wd = float.Parse(params_["wd"]);
            //var b1 = float.Parse(params_["beta1"]);
            //var b2 = float.Parse(params_["beta2"]);
            //var t = count_[index];
            //var coef1 = 1.0d - Math.Pow(b1, t);
            //var coef2 = 1.0d - Math.Pow(b2, t);
            //lr *= Math.Sqrt(coef2) / coef1;

            var inputs = new NDArrayHandle[4];
            inputs[0] = weight.GetHandle();
            inputs[1] = grad.GetHandle();
            inputs[2] = this._Mean[index].GetHandle();
            inputs[3] = this._Var[index].GetHandle();

            var numOutputs = 1;
            var output = weight.GetHandle();
            var outputs = new[] { output };

            NativeMethods.MXImperativeInvoke(this._UpdateHandle,
                                             4,
                                             inputs,
                                             ref numOutputs,
                                             ref outputs,
                                             keys.Length,
                                             keys,
                                             values);
        }

        #region Overrids

        protected override void CreateState(int index, NDArray weight)
        {
            this._Mean[index] = new NDArray(weight.GetShape());
            this._Mean[index].Set(0);
            this._Var[index] = new NDArray(weight.GetShape());
            this._Var[index].Set(0);
        }

        protected override void DisposeUnmanaged()
        {
            base.DisposeUnmanaged();

            foreach (var it in this._Mean)
                it.Value?.Dispose();

            foreach (var it in this._Var)
                it.Value?.Dispose();
        }

        #endregion

        #endregion

    }

}
