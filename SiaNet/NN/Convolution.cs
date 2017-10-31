using CNTK;
using System;
using SiaNet.Common;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet.NN
{
    public class Convolution
    {
        public static Function Conv1D(Variable layer, int channels, int kernalSize, int strides=1, bool padding=true, int dialation=1, string activation = OptActivations.None, bool useBias = false, string weightInitializer = OptInitializers.Xavier, string biasInitializer = OptInitializers.Zeros)
        {
            int numInputChannels = layer.Shape[layer.Shape.Rank - 1];
            var convParams = new Parameter(new int[] { kernalSize, numInputChannels, channels }, DataType.Float, Initializers.Get(weightInitializer), GlobalParameters.Device);
            var conv = CNTKLib.Convolution(convParams, layer, new int[] { strides }, new BoolVector(new bool[] { true }), new BoolVector(new bool[] { true }), new int[] { dialation });

            Parameter bias = null;
            if (useBias)
            {
                bias = new Parameter(conv.Output.Shape, DataType.Float, Initializers.Get(biasInitializer), GlobalParameters.Device);
                conv = CNTKLib.Plus(bias, conv);
            }
            
            return Basic.Activation(conv, activation);
        }

        public static Function Conv1D(Tuple<int, int> shape, int channels, int kernalSize, int strides = 1, bool padding = true, int dialation = 1, string activation = OptActivations.None, bool useBias = false, string weightInitializer = OptInitializers.Xavier, string biasInitializer = OptInitializers.Zeros)
        {
            Variable input = CNTKLib.InputVariable(new int[] { shape.Item1, shape.Item2 }, DataType.Float);
            return Conv1D(input, channels, kernalSize, strides, padding, dialation, activation, useBias, weightInitializer, biasInitializer);
        }

        public static Function Conv2D(Variable layer, int channels, Tuple<int, int> kernalSize, Tuple<int, int> strides, bool padding = true, Tuple<int, int> dialation = null, string activation = OptActivations.None, bool useBias = false, string weightInitializer = OptInitializers.Xavier, string biasInitializer = OptInitializers.Zeros)
        {
            int numInputChannels = layer.Shape[layer.Shape.Rank - 1];
            var convParams = new Parameter(new int[] { kernalSize.Item1, kernalSize.Item1, numInputChannels, channels }, DataType.Float, Initializers.Get(weightInitializer), GlobalParameters.Device);
            if (dialation == null)
            {
                dialation = new Tuple<int, int>(1, 1);
            }

            var conv = CNTKLib.Convolution(convParams, layer, new int[] { strides.Item1, strides.Item2 }, new BoolVector(new bool[] { true }), new BoolVector(new bool[] { true }), new int[] { dialation.Item1, dialation.Item2 });

            Parameter bias = null;
            if (useBias)
            {
                bias = new Parameter(conv.Output.Shape, DataType.Float, Initializers.Get(biasInitializer), GlobalParameters.Device);
                conv = CNTKLib.Plus(bias, conv);
            }

            return Basic.Activation(conv, activation);
        }

        public static Function Conv2D(Tuple<int, int, int> shape, int channels, Tuple<int, int> kernalSize, Tuple<int, int> strides, bool padding = true, Tuple<int, int> dialation = null, string activation = OptActivations.None, bool useBias = false, string weightInitializer = OptInitializers.Xavier, string biasInitializer = OptInitializers.Zeros)
        {
            Variable input = CNTKLib.InputVariable(new int[] { shape.Item1, shape.Item2, shape.Item3 }, DataType.Float);
            return Conv2D(input, channels, kernalSize, strides, padding, dialation, activation, useBias, weightInitializer, biasInitializer);
        }

        public static Function Conv3D(Variable layer, int channels, Tuple<int, int, int> kernalSize, Tuple<int, int, int> strides, bool padding = true, Tuple<int, int, int> dialation = null, string activation = OptActivations.None, bool useBias = false, string weightInitializer = OptInitializers.Xavier, string biasInitializer = OptInitializers.Zeros)
        {
            int numInputChannels = layer.Shape[layer.Shape.Rank - 1];
            var convParams = new Parameter(new int[] { kernalSize.Item1, kernalSize.Item2, kernalSize.Item3, numInputChannels, channels }, DataType.Float, Initializers.Get(weightInitializer), GlobalParameters.Device);

            if(dialation==null)
            {
                dialation = new Tuple<int, int, int>(1, 1, 1);
            }

            var conv = CNTKLib.Convolution(convParams, layer, new int[] { strides.Item1, strides.Item2, strides.Item3 }, new BoolVector(new bool[] { true }), new BoolVector(new bool[] { true }), new int[] { dialation.Item1, dialation.Item2, dialation.Item3 });
            Parameter bias = null;
            if (useBias)
            {
                bias = new Parameter(conv.Output.Shape, DataType.Float, Initializers.Get(biasInitializer), GlobalParameters.Device);
                conv = CNTKLib.Plus(bias, conv);
            }

            return Basic.Activation(conv, activation);
        }

        public static Function Conv3D(Tuple<int, int, int, int> shape, int channels, Tuple<int, int, int> kernalSize, Tuple<int, int, int> strides, bool padding = true, Tuple<int, int, int> dialation = null, string activation = OptActivations.None, bool useBias = false, string weightInitializer = OptInitializers.Xavier, string biasInitializer = OptInitializers.Zeros)
        {
            Variable input = CNTKLib.InputVariable(new int[] { shape.Item1, shape.Item2, shape.Item3 }, DataType.Float);
            return Conv3D(input, channels, kernalSize, strides, padding, dialation, activation, useBias, weightInitializer, biasInitializer);
        }

        public static Function MaxPool1D(Variable layer, int poolSize, int strides, bool padding=true)
        {
            return CNTKLib.Pooling(layer, PoolingType.Max, new int[] { poolSize }, new int[] { strides }, new BoolVector(new bool[] { true }));
        }

        public static Function MaxPool2D(Variable layer, Tuple<int, int> poolSize, Tuple<int, int> strides, bool padding = true)
        {
            return CNTKLib.Pooling(layer, PoolingType.Max, new int[] { poolSize.Item1, poolSize.Item2 }, new int[] { strides.Item1, strides.Item2 }, new BoolVector(new bool[] { true }));
        }

        public static Function MaxPool3D(Variable layer, Tuple<int, int, int> poolSize, Tuple<int, int, int> strides, bool padding = true)
        {
            return CNTKLib.Pooling(layer, PoolingType.Max, new int[] { poolSize.Item1, poolSize.Item2, poolSize.Item3 }, new int[] { strides.Item1, strides.Item2, strides.Item3 }, new BoolVector(new bool[] { true }));
        }

        public static Function AvgPool1D(Variable layer, int poolSize, int strides, bool padding = true)
        {
            return CNTKLib.Pooling(layer, PoolingType.Average, new int[] { poolSize }, new int[] { strides }, new BoolVector(new bool[] { true }));
        }

        public static Function AvgPool2D(Variable layer, Tuple<int, int> poolSize, Tuple<int, int> strides, bool padding = true)
        {
            return CNTKLib.Pooling(layer, PoolingType.Average, new int[] { poolSize.Item1, poolSize.Item2 }, new int[] { strides.Item1, strides.Item2 }, new BoolVector(new bool[] { true }));
        }

        public static Function AvgPool3D(Variable layer, Tuple<int, int, int> poolSize, Tuple<int, int, int> strides, bool padding = true)
        {
            return CNTKLib.Pooling(layer, PoolingType.Average, new int[] { poolSize.Item1, poolSize.Item2, poolSize.Item3 }, new int[] { strides.Item1, strides.Item2, strides.Item3 }, new BoolVector(new bool[] { true }));
        }

        public static Function GlobalMaxPool1D(Variable layer)
        {
            return CNTKLib.Pooling(layer, PoolingType.Max, new int[] { layer.Shape[0] });
        }

        public static Function GlobalMaxPool2D(Variable layer)
        {
            return CNTKLib.Pooling(layer, PoolingType.Max, new int[] { layer.Shape[0], layer.Shape[1] });
        }

        public static Function GlobalMaxPool3D(Variable layer)
        {
            return CNTKLib.Pooling(layer, PoolingType.Max, new int[] { layer.Shape[0], layer.Shape[1], layer.Shape[2] });
        }

        public static Function GlobalAvgPool1D(Variable layer)
        {
            return CNTKLib.Pooling(layer, PoolingType.Average, new int[] { layer.Shape[0] }); 
        }

        public static Function GlobalAvgPool2D(Variable layer)
        {
            return CNTKLib.Pooling(layer, PoolingType.Average, new int[] { layer.Shape[0], layer.Shape[1] });
        }

        public static Function GlobalAvgPool3D(Variable layer)
        {
            return CNTKLib.Pooling(layer, PoolingType.Average, new int[] { layer.Shape[0], layer.Shape[1], layer.Shape[2] });
        }
    }
}
