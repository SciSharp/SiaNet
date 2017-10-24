using System;
using System.Collections.Generic;
using System.Dynamic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet.Model.Layers
{
    public class Conv1D : LayerConfig
    {
        public Conv1D()
        {
            base.Name = "Conv1D";
            base.Params = new ExpandoObject();
        }

        public Conv1D(int channels, int kernalSize, int strides = 1, bool padding = true, int dialation = 1, string activation = OptActivations.None, bool useBias = false, string weightInitializer = OptInitializers.Xavier, string biasInitializer = OptInitializers.Zeros)
            : this()
        {
            Shape = null;
            Channels = channels;
            KernalSize = kernalSize;
            Padding = padding;
            Dialation = dialation;
            Act = activation;
            UseBias = useBias;
            WeightInitializer = weightInitializer;
            BiasInitializer = biasInitializer;
        }

        public Conv1D(Tuple<int, int> shape, int channels, int kernalSize, int strides = 1, bool padding = true, int dialation = 1, string activation = OptActivations.None, bool useBias = false, string weightInitializer = OptInitializers.Xavier, string biasInitializer = OptInitializers.Zeros)
            : this(channels, kernalSize, strides, padding, dialation,activation, useBias, weightInitializer, biasInitializer)
        {
            Shape = Tuple.Create<int, int>(shape.Item1, shape.Item2);
        }

        [Newtonsoft.Json.JsonIgnore]
        public Tuple<int, int> Shape
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
        public int Channels
        {
            get
            {
                return base.Params.Channels;
            }

            set
            {
                base.Params.Channels = value;
            }
        }

        [Newtonsoft.Json.JsonIgnore]
        public int KernalSize
        {
            get
            {
                return base.Params.KernalSize;
            }

            set
            {
                base.Params.KernalSize = value;
            }
        }

        [Newtonsoft.Json.JsonIgnore]
        public int Strides
        {
            get
            {
                return base.Params.Strides;
            }

            set
            {
                base.Params.Strides = value;
            }
        }

        [Newtonsoft.Json.JsonIgnore]
        public bool Padding
        {
            get
            {
                return base.Params.Padding;
            }

            set
            {
                base.Params.Padding = value;
            }
        }

        [Newtonsoft.Json.JsonIgnore]
        public int Dialation
        {
            get
            {
                return base.Params.Dialation;
            }

            set
            {
                base.Params.Dialation = value;
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
