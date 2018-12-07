using System;
using System.Collections.Generic;
using System.Globalization;
using SiaNet.Backend.Interop;
using NDArrayHandle = System.IntPtr;
using AtomicSymbolCreator = System.IntPtr;

// ReSharper disable once CheckNamespace
namespace SiaNet.Backend
{

    public sealed class RMSPropOptimizer : BaseOptimizer
    {

        #region Fields

        private readonly AtomicSymbolCreator _UpdateHandle;

        private readonly AtomicSymbolCreator _AlexUpdateHandle;

        private readonly Dictionary<int, NDArray> _N = new Dictionary<int, NDArray>();

        private readonly Dictionary<int, NDArray> _G = new Dictionary<int, NDArray>();

        private readonly Dictionary<int, NDArray> _Delta = new Dictionary<int, NDArray>();

        #endregion

        #region Constructors

        public RMSPropOptimizer()
            : this(0)
        {
        }

        public RMSPropOptimizer(uint beginNumUpdate)
            : base(beginNumUpdate)
        {
            this._UpdateHandle = OpMap.GetSymbolCreator("rmsprop_update");
            this._AlexUpdateHandle = OpMap.GetSymbolCreator("rmspropalex_update");
            this.SetParam("gamma1", 0.9f);
            this.SetParam("gamma2", 0.9f);
            this.SetParam("epsilon", 1e-8);
        }

        #endregion

        #region Methods

        public override string GetOptimizerType()
        {
            return "rmsprop";
        }

        public override void Update(int index, NDArray weight, NDArray grad)
        {
            if (weight == null)
                throw new ArgumentNullException(nameof(weight));
            if (grad == null)
                throw new ArgumentNullException(nameof(grad));

            if (!this._N.ContainsKey(index))
                this.CreateState(index, weight);

            this.Params["lr"] = this.GetLearningRate(index).ToString(CultureInfo.InvariantCulture);
            this.Params["wd"] = this.GetWeightDecay(index).ToString(CultureInfo.InvariantCulture);
            this.UpdateCount(index);
            var keys = this.GetParamKeys_();
            var values = this.GetParamValues_();
            Logging.CHECK_EQ(keys.Length, values.Length);

            var inputs = new NDArrayHandle[5];
            inputs[0] = weight.GetHandle();
            inputs[1] = grad.GetHandle();
            inputs[2] = this._N[index].GetHandle();
            inputs[3] = this._G[index].GetHandle();
            inputs[4] = this._Delta[index].GetHandle();

            var num_outputs = 1;
            var output = weight.GetHandle();
            var outputs = new[] { output };

            NativeMethods.MXImperativeInvoke(this._AlexUpdateHandle,
                                             5,
                                             inputs,
                                             ref num_outputs,
                                             ref outputs,
                                             keys.Length,
                                             keys,
                                             values);
        }

        #region Overrids

        protected override void CreateState(int index, NDArray weight)
        {
            this._N[index] = new NDArray(weight.GetShape());
            this._N[index].Set(0);
            this._G[index] = new NDArray(weight.GetShape());
            this._G[index].Set(0);
            this._Delta[index] = new NDArray(weight.GetShape());
            this._Delta[index].Set(0);
        }

        protected override void DisposeUnmanaged()
        {
            base.DisposeUnmanaged();

            foreach (var it in this._N)
                it.Value?.Dispose();

            foreach (var it in this._G)
                it.Value?.Dispose();

            foreach (var it in this._Delta)
                it.Value?.Dispose();
        }

        #endregion

        #endregion

    }

}
