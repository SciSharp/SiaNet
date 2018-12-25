using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Layers.Activations
{
    internal class ActivationRegistry
    {
        internal static BaseLayer Get(ActivationType activationType)
        {
            BaseLayer act = null;

            switch (activationType)
            {
                case ActivationType.ReLU:
                    act = new Relu();
                    break;
                case ActivationType.Sigmoid:
                    act = new Sigmoid();
                    break;
                case ActivationType.Tanh:
                    act = new Tanh();
                    break;
                case ActivationType.Elu:
                    act = new Elu();
                    break;
                case ActivationType.Exp:
                    act = new Exp();
                    break;
                case ActivationType.HargSigmoid:
                    act = new HardSigmoid();
                    break;
                case ActivationType.LeakyReLU:
                    act = new LeakyRelu();
                    break;
                case ActivationType.PReLU:
                    act = new PRelu();
                    break;
                case ActivationType.SeLU:
                    act = new Selu();
                    break;
                case ActivationType.Softmax:
                    act = new Softmax();
                    break;
                case ActivationType.Softplus:
                    act = new Softplus();
                    break;
                case ActivationType.SoftSign:
                    act = new Softsign();
                    break;
                default:
                    break;
            }

            return act;
        }
    }
}
