using System;
using System.Collections.Generic;
using System.Text;
using SiaNet.Backend;

namespace SiaNet.Layers
{
    public class Dropout : BaseLayer, ILayer
    {
        public float Rate { get; set; }

        public DropoutMode Mode { get; set; }

        public Dropout(float rate, DropoutMode mode = DropoutMode.Training)
            :base("dropout")
        {

        }

        public Symbol Build(Symbol data)
        {
            return Operators.Dropout(ID, data, Rate, Mode);
        }
        
    }
}
