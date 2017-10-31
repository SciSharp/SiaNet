using SiaNet.Common;
using System;
using System.Collections.Generic;
using System.Dynamic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet.Model.Layers
{
    public class Conv2D : LayerConfig
    {
        public Conv2D()
        {
            base.Name = "Conv2D";
            base.Params = new ExpandoObject();
        }

        public Conv2D(int channels, Tuple<int, int> kernalSize, Tuple<int, int> strides = null, bool padding = true, Tuple<int, int> dialation = null, string activation = OptActivations.None, bool useBias = false, string weightInitializer = OptInitializers.Xavier, string biasInitializer = OptInitializers.Zeros)
            : this()
        {
            Shape = null;
            Channels = channels;
            KernalSize = kernalSize;
            Strides = strides == null ? Tuple.Create<int, int>(1, 1) : strides;
            Padding = padding;
            Dialation = dialation == null ? Tuple.Create<int, int>(1, 1) : dialation;
            Act = activation;
            UseBias = useBias;
            WeightInitializer = weightInitializer;
            BiasInitializer = biasInitializer;
        }

        public Conv2D(Tuple<int, int, int> shape, int channels, Tuple<int, int> kernalSize, Tuple<int, int> strides = null, bool padding = true, Tuple<int, int> dialation = null, string activation = OptActivations.None, bool useBias = false, string weightInitializer = OptInitializers.Xavier, string biasInitializer = OptInitializers.Zeros)
            : this(channels, kernalSize, strides, padding, dialation, activation, useBias, weightInitializer, biasInitializer)
        {
            Shape = shape;
        }

        [Newtonsoft.Json.JsonIgnore]
        public Tuple<int, int, int> Shape
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
        public Tuple<int, int> KernalSize
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
        public Tuple<int, int> Strides
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
        public Tuple<int, int> Dialation
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
