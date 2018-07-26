using System;
using System.Collections.Generic;
using System.IO;
using CNTK;
using Newtonsoft.Json;
using SiaNet.Model;

namespace SiaNet
{
    /// <summary>
    ///     The Sequential model is a linear stack of layers.
    /// </summary>
    /// <seealso cref="SiaNet.Model.ConfigModule" />
    public class Sequential : ConfigModule
    {
        private readonly List<LayerBase> _layers = new List<LayerBase>();

        /// <summary>
        ///     Initializes a new instance of the <see cref="Sequential" /> class.
        /// </summary>
        public Sequential(Shape inputShape)
        {
            InputShape = inputShape;
        }

        public Shape InputShape { get; }

        /// <summary>
        ///     Gets or sets the stacked layers cofiguration.
        /// </summary>
        /// <value>The layers.</value>
        public LayerBase[] Layers
        {
            get => _layers.ToArray();
        }

        /// <summary>
        ///     Loads the neural network configuration saved using SaveLayers method.
        /// </summary>
        /// <param name="filepath">The filepath.</param>
        /// <returns>Sequential model</returns>
        public static Sequential LoadLayers(string filepath)
        {
            var json = File.ReadAllText(filepath);
            var result = JsonConvert.DeserializeObject<Sequential>(json, new JsonSerializerSettings
            {
                TypeNameHandling = TypeNameHandling.All
            });

            return result;
        }

        /// <summary>
        ///     Stack the neural layers for building a deep learning model.
        /// </summary>
        /// <param name="base">The configuration.</param>
        public void Add(LayerBase @base)
        {
            _layers.Add(@base);
        }

        /// <summary>
        ///     Configures the model for training.
        /// </summary>
        public CompiledModel Compile()
        {
            CNTK.Function compiledModel = null;
            foreach (var layer in _layers)
            {
                if (compiledModel == null)
                {
                    if (layer is OptimizableLayerBase)
                    {
                        compiledModel = (layer as OptimizableLayerBase).ToFunction(InputShape);
                    }
                    else
                    {
                        throw new Exception("First layer is not optimizable.");
                    }
                }
                else
                {
                    compiledModel = layer.ToFunction((Model.Function)compiledModel);
                }
            }

            return new CompiledModel(compiledModel);
        }

        /// <summary>
        ///     Saves the neural network configuration as json file.
        /// </summary>
        /// <param name="filepath">The filepath.</param>
        public void SaveLayers(string filepath)
        {
            var json = JsonConvert.SerializeObject(this, Formatting.Indented, new JsonSerializerSettings
            {
                TypeNameHandling = TypeNameHandling.All,
                TypeNameAssemblyFormatHandling = TypeNameAssemblyFormatHandling.Simple
            });
            File.WriteAllText(filepath, json);
        }
    }
}