using System;
using System.Collections.Generic;

// ReSharper disable once CheckNamespace
namespace SiaNet.Backend
{

    internal static class OptimizerRegistry
    {

        #region Fields

        private static readonly Dictionary<string, Func<BaseOptimizer>> cmap = new Dictionary<string, Func<BaseOptimizer>>();

        #endregion

        #region Methods

        public static BaseOptimizer Find(string name)
        {
            MXNETCPP_REGISTER_OPTIMIZER<SGDOptimizer>("sgd");
            MXNETCPP_REGISTER_OPTIMIZER<SGDOptimizer>("ccsgd");  // For backward compatibility
            MXNETCPP_REGISTER_OPTIMIZER<RMSPropOptimizer>("rmsprop");
            MXNETCPP_REGISTER_OPTIMIZER<AdamOptimizer>("adam");
            MXNETCPP_REGISTER_OPTIMIZER<AdaGradOptimizer>("adagrad");
            MXNETCPP_REGISTER_OPTIMIZER<AdaDeltaOptimizer>("adadelta");
            MXNETCPP_REGISTER_OPTIMIZER<SignumOptimizer>("signum");

            return !cmap.TryGetValue(name, out var value) ? null : value.Invoke();
        }

        public static int Register(string name, Func<BaseOptimizer> creator)
        {
            Logging.CHECK_EQ(cmap.ContainsKey(name), false, " already registered");
            cmap.Add(name, creator);
            return 0;
        }

        #region Helpers

        private static void MXNETCPP_REGISTER_OPTIMIZER<T>(string name)
            where T : BaseOptimizer, new()
        {
            Register(name, () => new T());
        }

        #endregion

        #endregion

    }

}