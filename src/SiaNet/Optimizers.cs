using SiaNet.Backend;
using SiaNet;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet
{
    public class Optimizers
    {
        public static BaseOptimizer SGD(float lr = 0.01f, float momentum = float.Epsilon, float decay = 0)
        {
            var opt = OptimizerRegistry.Find("sgd");
            opt.SetParam("lr", lr);
            opt.SetParam("wd", decay);
            opt.SetParam("momentum", momentum);

            return opt;
        }

        public static BaseOptimizer Signum(float lr = 0.01f, float decay = 0, float momentum = float.Epsilon)
        {
            var opt = OptimizerRegistry.Find("signum");
            opt.SetParam("lr", lr);
            opt.SetParam("wd", decay);
            opt.SetParam("momentum", momentum);

            return opt;
        }

        public static BaseOptimizer RMSprop(float lr = 0.01f, float rho = 0.95f, float epsilon = float.Epsilon, float decay = 0)
        {
            var opt = OptimizerRegistry.Find("rmsprop");
            opt.SetParam("lr", lr);
            opt.SetParam("wd", decay);
            opt.SetParam("gamma1", rho);
            opt.SetParam("epsilon", epsilon);

            return opt;
        }

        public static BaseOptimizer Adagrad(float lr = 0.01f, float epsilon = float.Epsilon, float decay = 0)
        {
            var opt = OptimizerRegistry.Find("adagrad");
            opt.SetParam("lr", lr);
            opt.SetParam("wd", decay);
            opt.SetParam("eps", epsilon);

            return opt;
        }

        public static BaseOptimizer Adadelta(float lr = 1.0f, float rho = 0.95f, float epsilon = float.Epsilon, float decay = 0)
        {
            var opt = OptimizerRegistry.Find("adadelta");
            opt.SetParam("lr", lr);
            opt.SetParam("wd", decay);
            opt.SetParam("rho", rho);
            opt.SetParam("epsilon", epsilon);

            return opt;
        }

        public static BaseOptimizer Adam(float lr = 0.001f, float beta_1 = 0.9f, float beta_2 = 0.999f, float decay = 0, bool amsgrad = false)
        {
            var opt = OptimizerRegistry.Find("adam");
            opt.SetParam("lr", lr);
            opt.SetParam("wd", decay);
            opt.SetParam("beta1", beta_1);
            opt.SetParam("beta2", beta_2);
            opt.SetParam("epsilon", 1e-8);

            return opt;
        }

        internal static BaseOptimizer Get(OptimizerType optimizerType)
        {
            BaseOptimizer opt = null;
            switch (optimizerType)
            {
                case OptimizerType.SGD:
                    opt = SGD();
                    break;
                case OptimizerType.Signum:
                    opt = Signum();
                    break;
                case OptimizerType.RMSprop:
                    opt = RMSprop();
                    break;
                case OptimizerType.Adagrad:
                    opt = Adagrad();
                    break;
                case OptimizerType.Adadelta:
                    opt = Adadelta();
                    break;
                case OptimizerType.Adam:
                    opt = Adam();
                    break;
                default:
                    break;
            }

            return opt;
        }
    }
}
