using System;
using System.Collections.Generic;
using System.Dynamic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet.Model.Layers
{
    /// <summary>
    /// Dropout consists in randomly setting a fraction rate of input units to 0 at each update during training time, which helps prevent overfitting.
    /// </summary>
    /// <seealso cref="SiaNet.Model.LayerConfig" />
    public class Dropout : LayerConfig
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Dropout"/> class.
        /// </summary>
        public Dropout()
        {
            base.Name = "Dropout";
            base.Params = new ExpandoObject();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Dropout"/> class.
        /// </summary>
        /// <param name="rate">A float value between 0 and 1. Fraction of the input units to drop.</param>
        public Dropout(double rate)
            : this()
        {
            Rate = rate;
        }

        /// <summary>
        /// A float value between 0 and 1. Fraction of the input units to drop.
        /// </summary>
        /// <value>
        /// The rate.
        /// </value>
        [Newtonsoft.Json.JsonIgnore]
        public double Rate
        {
            get
            {
                return base.Params.Rate;
            }

            set
            {
                base.Params.Rate = value;
            }
        }

    }
}
