using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;

namespace SiaNet.Layers
{
    public abstract class BaseLayer
    {
        public Dictionary<string, Variable> Params { get; set; }

        public Variable Input { get; set; }

        public Variable Output { get; set; }

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
                return Params[name];
            }
            set
            {
                Params[name] = value;
            }
        }
    }
}
