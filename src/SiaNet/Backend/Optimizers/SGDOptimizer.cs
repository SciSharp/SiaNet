using System;
using System.Collections.Generic;
using System.Globalization;
using SiaNet.Backend.Interop;
using NDArrayHandle = System.IntPtr;
using AtomicSymbolCreator = System.IntPtr;

// ReSharper disable once CheckNamespace
namespace SiaNet.Backend
{

    public sealed class SGDOptimizer : BaseOptimizer
    {

        #region Fields

        private readonly Dictionary<int, NDArray> _States = new Dictionary<int, NDArray>();

        private readonly AtomicSymbolCreator _UpdateHandle;

        private readonly AtomicSymbolCreator _MomUpdateHandle;

        #endregion

        #region Constructors

        public SGDOptimizer()
            : this(0)
        {

        }

        public SGDOptimizer(uint beginNumUpdate)
            : base(beginNumUpdate)
        {
            this._UpdateHandle = OpMap.GetSymbolCreator("sgd_update");
            this._MomUpdateHandle = OpMap.GetSymbolCreator("sgd_mom_update");
        }

        #endregion

        #region Methods

        public override string GetOptimizerType()
        {
            return "sgd";
        }

        public override void Update(int index, NDArray weight, NDArray grad)
        {
            if (weight == null)
                throw new ArgumentNullException(nameof(weight));
            if (grad == null)
                throw new ArgumentNullException(nameof(grad));

            if (!this._States.ContainsKey(index))
                this.CreateState(index, weight);

            this.Params["lr"] = this.GetLearningRate(index).ToString(CultureInfo.InvariantCulture);
            this.Params["wd"] = this.GetWeightDecay(index).ToString(CultureInfo.InvariantCulture);
            this.UpdateCount(index);
            var keys = this.GetParamKeys_();
            var values = this.GetParamValues_();
            Logging.CHECK_EQ(keys.Length, values.Length);

            var inputs = new NDArrayHandle[3];
            inputs[0] = weight.GetHandle();
            inputs[1] = grad.GetHandle();

            var numOutputs = 1;
            var output = weight.GetHandle();
            var outputs = new[] { output };

            if (this._States[index] == null)
            {
                NativeMethods.MXImperativeInvoke(this._UpdateHandle,
                                                 2,
                                                 inputs,
                                                 ref numOutputs,
                                                 ref outputs,
                                                 keys.Length,
                                                 keys,
                                                 values);
            }
            else
            {
                inputs[2] = this._States[index].GetHandle();
                NativeMethods.MXImperativeInvoke(this._MomUpdateHandle,
                                                 3,
                                                 inputs,
                                                 ref numOutputs,
                                                 ref outputs,
                                                 keys.Length,
                                                 keys,
                                                 values);
            }
        }

        #region Overrids

        protected override void DisposeUnmanaged()
        {
            base.DisposeUnmanaged();
            foreach (var it in this._States)
                it.Value?.Dispose();
        }

        protected override void CreateState(int index, NDArray weight)
        {
            if (!this.Params.ContainsKey("momentum"))
            {
                this._States[index] = null;
            }
            else
            {
                this._States[index] = new NDArray(weight.GetShape());
                this._States[index].Set(0f);
            }
        }

        #endregion

        #endregion

    }

}
