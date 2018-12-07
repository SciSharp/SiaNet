using SiaDNN.Constraints;
using SiaDNN.Initializers;
using SiaNet.Backend;
using SiaNet.Regularizers;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Layers
{
    public abstract class BaseLayer
    {
        public string Name { get; set; }

        public string ID { get; set; }

        public Dictionary<string, BaseInitializer> InitParams;
        public Dictionary<string, BaseConstraint> ConstraintParams;
        public Dictionary<string, BaseRegularizer> RegularizerParams;

        public BaseLayer(string name)
        {
            Name = name;
            ID = UUID.GetID(name);
            InitParams = new Dictionary<string, BaseInitializer>();
            ConstraintParams = new Dictionary<string, BaseConstraint>();
            RegularizerParams = new Dictionary<string, BaseRegularizer>();
        }
    }
}
