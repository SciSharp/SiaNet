using SiaNet.Constraints;
using SiaNet.Engine;
using SiaNet.Initializers;
using SiaNet.Regularizers;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Layers
{
    public abstract class BaseLayer
    {
        internal IBackend K = Global.CurrentBackend;

        [NonSerialized]
        public Dictionary<string, Parameter> Params;

        [NonSerialized]
        public Parameter Input;

        [NonSerialized]
        public Tensor Output;

        public string Name { get; set; }

        public bool SkipPred { get; set; }

        public BaseLayer(string name)
        {
            Name = K.UUID(name);
            Params = new Dictionary<string, Parameter>();
        }

        public virtual void Forward(Tensor x)
        {
            Input = x.ToParameter();
        }

        public virtual void Backward(Tensor outputgrad)
        {

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

        public Parameter BuildParam(string name, long[] shape, DataType elementType, BaseInitializer initializer, BaseConstraint constraint = null, BaseRegularizer regularizer = null, bool trainable = true)
        {
            Parameter v = null;
            name = Name + "_" + name;
            if (!Params.ContainsKey(name))
            {
                v = new Parameter(name, elementType, shape);
                v.Data = initializer.Operator(shape);
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
