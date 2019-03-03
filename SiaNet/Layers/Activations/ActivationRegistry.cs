using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Layers.Activations
{
    internal class ActivationRegistry
    {
        internal static BaseLayer Get(ActType activationType)
        {
            BaseLayer act = null;

            switch (activationType)
            {
                case ActType.ReLU:
                    act = new Relu();
                    break;
                case ActType.Sigmoid:
                    act = new Sigmoid();
                    break;
                case ActType.Tanh:
                    act = new Tanh();
                    break;
                case ActType.Elu:
                    act = new Elu();
                    break;
                case ActType.Exp:
                    act = new Exp();
                    break;
                case ActType.HargSigmoid:
                    act = new HardSigmoid();
                    break;
                case ActType.LeakyReLU:
                    act = new LeakyRelu();
                    break;
                case ActType.PReLU:
                    act = new PRelu();
                    break;
                case ActType.SeLU:
                    act = new Selu();
                    break;
                case ActType.Softmax:
                    act = new Softmax();
                    break;
                case ActType.Softplus:
                    act = new Softplus();
                    break;
                case ActType.SoftSign:
                    act = new Softsign();
                    break;
                case ActType.Linear:
                    act = new Linear();
                    break;
                default:
                    break;
            }

            return act;
        }
    }
}
