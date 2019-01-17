using SiaNet.Constraints;
using SiaNet.Initializers;
using SiaNet.Regularizers;
using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;

namespace SiaNet.Layers
{
    public abstract class BaseLayer : TOps
    {
        public Dictionary<string, Parameter> Params { get; set; }

        public Parameter Input { get; set; }

        public Tensor Output { get; set; }

        public string Name { get; set; }

        public BaseLayer(string name)
        {
            Name = UUID.GetID(name);
            Params = new Dictionary<string, Parameter>();
        }

        public abstract void Forward(Parameter x);

        public abstract void Backward(Tensor outputgrad);

        public IEnumerable<Parameter> GetParameters()
        {
            foreach (var item in Params)
            {
                yield return item.Value;
            }
        }
        
        public Parameter this[string name]
        {
            get
            {
                return Params[Name + "_" +name];
            }
            set
            {
                Params[Name + "_" + name] = value;
            }
        }

        public Parameter BuildParam(string name, long[] shape, DType elementType, BaseInitializer initializer, BaseConstraint constraint = null, BaseRegularizer regularizer = null, bool trainable = true)
        {
            Parameter v = null;
            name = Name + "_" + name;
            if (!Params.ContainsKey(name))
            {
                v = new Parameter(name, elementType, shape);
                v.Data = initializer.Operator(v.Data.Shape);
                v.SetConstraint(constraint);
                v.SetRegularizer(regularizer);
                if(trainable)
                    Params.Add(name, v);
            }
            else
            {
                v = Params[name];
            }

            return v;
        }
    }
}
