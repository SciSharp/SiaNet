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
        public Dictionary<string, Variable> Params { get; set; }

        public Variable Input { get; set; }

        public Tensor Output { get; set; }

        public string Name { get; set; }

        public BaseLayer(string name)
        {
            Name = UUID.GetID(name);
            Params = new Dictionary<string, Variable>();
        }

        public abstract void Forward(Variable x);

        public abstract void Backward(Tensor outputgrad);

        public IEnumerable<Variable> GetParameters()
        {
            foreach (var item in Params)
            {
                yield return item.Value;
            }
        }
        
        public Variable this[string name]
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

        public Variable BuildVar(string name, long[] shape, DType elementType, BaseInitializer initializer, BaseConstraint constraint = null, BaseRegularizer regularizer = null, bool trainable = true)
        {
            Variable v = null;
            name = Name + "_" + name;
            if (!Params.ContainsKey(name))
            {
                v = new Variable(name, elementType, shape);
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
