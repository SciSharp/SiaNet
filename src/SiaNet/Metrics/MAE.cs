using SiaNet.Backend;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Metrics
{
    public sealed class MAE : BaseMetric
    {
        #region Constructors

        public MAE() : base("mae") { }

        #endregion

        #region Methods

        public override void Update(NDArray labels, NDArray preds)
        {
            if (labels == null)
                throw new ArgumentNullException(nameof(labels));
            if (preds == null)
                throw new ArgumentNullException(nameof(preds));

            //Logging.CHECK_EQ(labels.GetShape().Count, preds.GetShape().Count);

            preds = preds.Reshape(new Shape(preds.GetShape()[0]));
            var result = NDArray.Mean(NDArray.Abs(preds - labels)).AsArray();

            this.SumMetric += result.Length > 0 ? result[0] : 0;
            this.NumInst += 1;
        }

        #endregion

    }
}
