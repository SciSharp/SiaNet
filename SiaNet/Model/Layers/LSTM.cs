using System;
using System.Collections.Generic;
using CNTK;
using Newtonsoft.Json;
using SiaNet.Model.Initializers;
using SiaNet.Model.Layers.Activations;

namespace SiaNet.Model.Layers
{
    /// <summary>
    ///     Long short-term memory (LSTM) is a recurrent neural network (RNN) architecture that remembers values over arbitrary
    ///     intervals
    /// </summary>
    /// <seealso cref="OptimizableLayerBase" />
    public class LSTM : OptimizableLayerBase
    {
        /// <summary>
        ///     Initializes a new instance of the <see cref="LSTM" /> class.
        /// </summary>
        /// <param name="dim">Positive integer, dimensionality of the output space.</param>
        /// <param name="shape">The input shape.</param>
        /// <param name="activation">
        ///     Activation function to use. If you don't specify anything, no activation is applied (ie.
        ///     "linear" activation: a(x) = x). <see cref="SiaNet.Common.OptActivations" />
        /// </param>
        /// <param name="useBias">Boolean, whether the layer uses a bias vector.</param>
        /// <param name="weightInitializer">
        ///     Initializer for the kernel weights matrix. <see cref="SiaNet.Common.OptInitializers" />
        /// </param>
        /// <param name="biasInitializer">Initializer for the bias vector. <see cref="SiaNet.Common.OptInitializers" /></param>
        public LSTM(
            int dim,
            int? cellDim = null,
            ActivationBase activation = null,
            ActivationBase recurrentActivation = null,
            InitializerBase weightInitializer = null,
            InitializerBase recurrentInitializer = null,
            bool useBias = true,
            InitializerBase biasInitializer = null,
            bool returnSequence = false)
        {
            Dim = dim;
            CellDim = cellDim;
            Activation = activation ?? new Tanh();
            RecurrentActivation = recurrentActivation ?? new Sigmoid();
            UseBias = useBias;
            ReturnSequence = returnSequence;
            WeightInitializer = weightInitializer ?? new GlorotUniform();
            RecurrentInitializer = recurrentInitializer ?? new GlorotUniform();
            BiasInitializer = biasInitializer ?? new Zeros();
        }


        /// <summary>
        ///     Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x)
        ///     = x)
        /// </summary>
        /// <value>
        ///     The activation function.
        /// </value>
        [JsonIgnore]
        public ActivationBase Activation
        {
            get => GetParam<ActivationBase>("Activation");

            set => SetParam("Activation", value);
        }

        /// <summary>
        ///     Initializer for the bias vector.
        /// </summary>
        /// <value>
        ///     The bias initializer.
        /// </value>
        [JsonIgnore]
        public InitializerBase BiasInitializer
        {
            get => GetParam<InitializerBase>("BiasInitializer");

            set => SetParam("BiasInitializer", value);
        }

        public int? CellDim
        {
            get => GetParam<int?>("CellDim");

            set => SetParam("CellDim", value);
        }

        /// <summary>
        ///     Positive integer, dimensionality of the output space.
        /// </summary>
        /// <value>
        ///     The output dimension.
        /// </value>
        [JsonIgnore]
        public int Dim
        {
            get => GetParam<int>("Dim");

            set => SetParam("Dim", value);
        }

        [JsonIgnore]
        public ActivationBase RecurrentActivation
        {
            get => GetParam<ActivationBase>("RecurrentActivation");

            set => SetParam("RecurrentActivation", value);
        }

        [JsonIgnore]
        public InitializerBase RecurrentInitializer
        {
            get => GetParam<InitializerBase>("RecurrentInitializer");

            set => SetParam("RecurrentInitializer", value);
        }

        /// <summary>
        ///     Gets or sets a value indicating whether [return sequence].
        /// </summary>
        /// <value>
        ///     <c>true</c> if [return sequence]; otherwise, <c>false</c>.
        /// </value>
        [JsonIgnore]
        public bool ReturnSequence
        {
            get => GetParam<bool>("ReturnSequence");

            set => SetParam("ReturnSequence", value);
        }

        /// <summary>
        ///     Boolean, whether the layer uses a bias vector.
        /// </summary>
        /// <value>
        ///     <c>true</c> if [use bias]; otherwise, <c>false</c>.
        /// </value>
        [JsonIgnore]
        public bool UseBias
        {
            get => GetParam<bool>("UseBias");

            set => SetParam("UseBias", value);
        }

        /// <summary>
        ///     Initializer for the kernel weights matrix .
        /// </summary>
        /// <value>
        ///     The weight initializer.
        /// </value>
        [JsonIgnore]
        public InitializerBase WeightInitializer
        {
            get => GetParam<InitializerBase>("WeightInitializer");

            set => SetParam("WeightInitializer", value);
        }

        /// <inheritdoc />
        internal override Function ToFunction(Variable inputFunction)
        {
            //if (inputFunction.Shape.Rank < 3)
            //{
            //    throw new ArgumentException("Variable has an invalid shape.", nameof(inputFunction));
            //}

            var cellDim = CellDim.HasValue ? CellDim : Dim;
            var prevOutput =
                CNTK.Variable.PlaceholderVariable(new[] {Dim}, ((CNTK.Variable) inputFunction).DynamicAxes);
            var prevCellState = cellDim.HasValue
                ? CNTK.Variable.PlaceholderVariable(new[] {cellDim.Value}, ((CNTK.Variable) inputFunction).DynamicAxes)
                : null;

            Func<int, CNTK.Parameter> createBiasParam = d =>
                new CNTK.Parameter(new[] {d}, DataType.Float, BiasInitializer.ToDictionary(), GlobalParameters.Device);

            Func<int, CNTK.Parameter> createProjectionParam = oDim => new CNTK.Parameter(
                new[] {oDim, NDShape.InferredDimension},
                DataType.Float, WeightInitializer.ToDictionary(), GlobalParameters.Device);

            Func<int, CNTK.Parameter> createDiagWeightParam = d =>
                new CNTK.Parameter(new[] {d}, DataType.Float, RecurrentInitializer.ToDictionary(),
                    GlobalParameters.Device);

            var stabilizedPrevOutput = Stabilize<float>(prevOutput, GlobalParameters.Device);
            var stabilizedPrevCellState =
                prevCellState != null ? Stabilize<float>(prevCellState, GlobalParameters.Device) : null;

            Func<CNTK.Variable> projectInput = null;

            if (cellDim.HasValue)
            {
                projectInput = () => createBiasParam(cellDim.Value) +
                                     createProjectionParam(cellDim.Value) * (CNTK.Variable) inputFunction;
            }
            else
            {
                projectInput = () => inputFunction;
            }

            //Input gate
            CNTK.Function it = null;

            if (cellDim.HasValue)
            {
                it = RecurrentActivation.ToFunction(
                    (Function) ((CNTK.Variable) (projectInput() +
                                                 createProjectionParam(cellDim.Value) * stabilizedPrevOutput) +
                                CNTKLib.ElementTimes(createDiagWeightParam(cellDim.Value),
                                    stabilizedPrevCellState)));
            }
            else
            {
                it = RecurrentActivation.ToFunction(projectInput());
            }

            CNTK.Function bit = null;

            if (cellDim.HasValue)
            {
                bit = CNTKLib.ElementTimes(it,
                    Activation.ToFunction(
                        (Function) (projectInput() +
                                    createProjectionParam(cellDim.Value) * stabilizedPrevOutput)));
            }
            else
            {
                bit = CNTKLib.ElementTimes(it, Activation.ToFunction(projectInput()));
            }

            // Forget-me-not gate
            CNTK.Function ft = null;

            if (cellDim.HasValue)
            {
                ft = RecurrentActivation.ToFunction((Function) (
                    (CNTK.Variable) (projectInput() + createProjectionParam(cellDim.Value) * stabilizedPrevOutput) +
                    CNTKLib.ElementTimes(createDiagWeightParam(cellDim.Value), stabilizedPrevCellState)));
            }
            else
            {
                ft = RecurrentActivation.ToFunction(projectInput());
            }

            var bft = prevCellState != null ? CNTKLib.ElementTimes(ft, prevCellState) : ft;

            var ct = (CNTK.Variable) bft + bit;

            //Output gate
            CNTK.Function ot = null;

            if (cellDim.HasValue)
            {
                ot = RecurrentActivation.ToFunction((Function) (
                    (CNTK.Variable) (projectInput() + createProjectionParam(cellDim.Value) * stabilizedPrevOutput) +
                    CNTKLib.ElementTimes(createDiagWeightParam(cellDim.Value),
                        Stabilize<float>(ct, GlobalParameters.Device))));
            }
            else
            {
                ot = RecurrentActivation.ToFunction(
                    (Function) (projectInput() + Stabilize<float>(ct, GlobalParameters.Device)));
            }

            var ht = CNTKLib.ElementTimes(ot, CNTKLib.Tanh(ct));
            var c = ct;
            var h = Dim != cellDim ? createProjectionParam(Dim) * Stabilize<float>(ht, GlobalParameters.Device) : ht;

            Func<CNTK.Variable, CNTK.Function> recurrenceHookH = x => CNTKLib.PastValue(x);
            Func<CNTK.Variable, CNTK.Function> recurrenceHookC = x => CNTKLib.PastValue(x);

            var actualDh = recurrenceHookH(h);
            var actualDc = recurrenceHookC(c);

            if (prevCellState != null)
            {
                h.ReplacePlaceholders(
                    new Dictionary<CNTK.Variable, CNTK.Variable> {{prevOutput, actualDh}, {prevCellState, actualDc}});
            }
            else
            {
                h.ReplacePlaceholders(new Dictionary<CNTK.Variable, CNTK.Variable> {{prevOutput, actualDh}});
            }

            if (ReturnSequence)
            {
                return h;
            }

            return CNTKLib.SequenceLast(h);
        }

        
        private CNTK.Function Stabilize<TElementType>(CNTK.Variable x, DeviceDescriptor device)
        {
            CNTK.Constant f, fInv;

            if (typeof(TElementType) == typeof(float))
            {
                f = CNTK.Constant.Scalar(4.0f, device);
                fInv = CNTK.Constant.Scalar(f.DataType, 1.0 / 4.0f);
            }
            else
            {
                f = CNTK.Constant.Scalar(4.0, device);
                fInv = CNTK.Constant.Scalar(f.DataType, 1.0 / 4.0f);
            }

            var beta = CNTKLib.ElementTimes(
                fInv,
                CNTKLib.Log(
                    CNTK.Constant.Scalar(f.DataType, 1.0) +
                    CNTKLib.Exp(CNTKLib.ElementTimes(f,
                        new CNTK.Parameter(new NDShape(), f.DataType, 0.99537863 /* 1/f*ln (e^f-1) */, device)))));

            return CNTKLib.ElementTimes(beta, x);
        }
    }
}