using System;
using System.Collections.Generic;
using System.Dynamic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SiaNet.Common;

namespace SiaNet.Model.Layers
{
    public class Dense : LayerConfig
    {
        public Dense()
        {
            base.Name = "Dense";
            base.Params = new ExpandoObject();
        }

        public Dense(int dim, string act = OptActivations.None, bool useBias = false, string weightInitializer = OptInitializers.Xavier, string biasInitializer = OptInitializers.Zeros)
            : this()
        {
            Shape = null;
            Dim = dim;
            Act = act;
            UseBias = useBias;
            WeightInitializer = weightInitializer;
            BiasInitializer = biasInitializer;
        }

        public Dense(int dim, int shape, string act = OptActivations.None, bool useBias = false, string weightInitializer = OptInitializers.Xavier, string biasInitializer = OptInitializers.Zeros)
            : this(dim, act, useBias, weightInitializer, biasInitializer)
        {
            Shape = shape;
        }

        [Newtonsoft.Json.JsonIgnore]
        public int? Shape
        {
            get
            {
                return base.Params.Shape;
            }

            set
            {
                base.Params.Shape = value;
            }
        }

        [Newtonsoft.Json.JsonIgnore]
        public int Dim
        {
            get
            {
                return base.Params.Dim;
            }

            set
            {
                base.Params.Dim = value;
            }
        }

        [Newtonsoft.Json.JsonIgnore]
        public string Act
        {
            get
            {
                return base.Params.Act;
            }

            set
            {
                base.Params.Act = value;
            }
        }

        [Newtonsoft.Json.JsonIgnore]
        public bool UseBias
        {
            get
            {
                return base.Params.UseBias;
            }

            set
            {
                base.Params.UseBias = value;
            }
        }

        [Newtonsoft.Json.JsonIgnore]
        public string WeightInitializer
        {
            get
            {
                return base.Params.WeightInitializer;
            }

            set
            {
                base.Params.WeightInitializer = value;
            }
        }

        [Newtonsoft.Json.JsonIgnore]
        public string BiasInitializer
        {
            get
            {
                return base.Params.BiasInitializer;
            }

            set
            {
                base.Params.BiasInitializer = value;
            }
        }
    }
}
