using CNTK;
using SiaNet.Common;
using SiaNet.Interface;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet.Model.Optimizers
{
    public class BaseOptimizer
    {
        public BaseOptimizer(string name)
        {
            Name = name;
            AdditionalParams = new Dictionary<string, object>();
        }

        public BaseOptimizer(string name, double lr)
        {
            Name = name;
            LearningRate = lr;
            AdditionalParams = new Dictionary<string, object>();
        }

        internal string Name { get; set; }

        internal double LearningRate { get; set; }

        internal Regulizers Regulizer { get; set; }

        internal Dictionary<string, object> AdditionalParams { get; set; }

        internal Learner GetDefault(Function model, Regulizers reg)
        {
            Regulizer = reg;
            switch (Name.Trim().ToLower())
            {
                case OptOptimizers.AdaDelta:
                    return AdaDelta(model, regulizer: Regulizer);
                case OptOptimizers.AdaGrad:
                    return AdaGrad(model, regulizer: Regulizer);
                case OptOptimizers.Adam:
                    return Adam(model, regulizer: Regulizer);
                case OptOptimizers.Adamax:
                    return Adamax(model, regulizer: Regulizer);
                case OptOptimizers.MomentumSGD:
                    return MomentumSGD(model, regulizer: Regulizer);
                case OptOptimizers.RMSProp:
                    return RMSprop(model, regulizer: Regulizer);
                case OptOptimizers.SGD:
                    return SGD(model, regulizer: Regulizer);
                default:
                    throw new NotImplementedException(string.Format("{0} is not implemented", Name));
            }
        }

        internal Learner Get(Function model, Regulizers reg)
        {
            Regulizer = reg;
            switch (Name.Trim().ToLower())
            {
                case OptOptimizers.AdaDelta:
                    return AdaDelta(model, LearningRate, (double)AdditionalParams["Rho"], (double)AdditionalParams["Epsilon"], regulizer: Regulizer);
                case OptOptimizers.AdaGrad:
                    return AdaGrad(model, LearningRate, regulizer: Regulizer);
                case OptOptimizers.Adam:
                    return Adam(model, LearningRate, (double)AdditionalParams["Momentum"], (double)AdditionalParams["VarianceMomentum"], (bool)AdditionalParams["UnitGain"], (double)AdditionalParams["Epsilon"], regulizer: Regulizer);
                case OptOptimizers.Adamax:
                    return Adamax(model, LearningRate, (double)AdditionalParams["Momentum"], (double)AdditionalParams["VarianceMomentum"], (bool)AdditionalParams["UnitGain"], (double)AdditionalParams["Epsilon"], regulizer: Regulizer);
                case OptOptimizers.MomentumSGD:
                    return MomentumSGD(model, LearningRate, (double)AdditionalParams["Momentum"], (bool)AdditionalParams["UnitGain"], regulizer: Regulizer);
                case OptOptimizers.RMSProp:
                    return RMSprop(model, LearningRate, (double)AdditionalParams["Gamma"], (double)AdditionalParams["Increment"], (double)AdditionalParams["Decrement"], (double)AdditionalParams["Min"], (double)AdditionalParams["Max"], Regulizer);
                case OptOptimizers.SGD:
                    return SGD(model, LearningRate, regulizer: Regulizer);
                default:
                    throw new NotImplementedException(string.Format("{0} is not implemented", Name));
            }
        }

        private AdditionalLearningOptions GetAdditionalLearningOptions()
        {
            AdditionalLearningOptions options = new AdditionalLearningOptions();
            if (Regulizer != null)
            {
                if (Regulizer.IsL1)
                    options.l1RegularizationWeight = Regulizer.L1;
                if (Regulizer.IsL2)
                    options.l1RegularizationWeight = Regulizer.L2;
            }

            return options;
        }

        /// <summary>
        /// SGD is an optimisation technique. It is an alternative to Standard Gradient Descent and other approaches like batch training or BFGS. It still leads to fast convergence, with some advantages:
        /// - Doesn't require storing all training data in memory (good for large training sets)
        /// - Allows adding new data in an "online" setting
        /// </summary>
        /// <param name="modelOutput">The model output.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="regulizer">The regulizer.</param>
        /// <returns>Learner.</returns>
        private Learner SGD(Function modelOutput, double learningRate = 0.01, Regulizers regulizer = null)
        {
            CNTK.TrainingParameterScheduleDouble learningRatePerSample = new CNTK.TrainingParameterScheduleDouble(learningRate);
            return CNTKLib.SGDLearner(new ParameterVector(modelOutput.Parameters().ToList()), learningRatePerSample, GetAdditionalLearningOptions());
        }

        /// <summary>
        /// Momentum of Stochastic gradient descent optimizer.
        /// </summary>
        /// <param name="modelOutput">The model output.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="momentum">The momentum.</param>
        /// <param name="unitGain">if set to <c>true</c> [unit gain].</param>
        /// <param name="regulizer">The regulizer.</param>
        /// <returns>Learner.</returns>
        private Learner MomentumSGD(Function modelOutput, double learningRate = 0.01, double momentum = 0.1, bool unitGain = true, Regulizers regulizer = null)
        {
            CNTK.TrainingParameterScheduleDouble learningRatePerSample = new CNTK.TrainingParameterScheduleDouble(learningRate);
            CNTK.TrainingParameterScheduleDouble momentumPerSample = new CNTK.TrainingParameterScheduleDouble(momentum);

            return CNTKLib.MomentumSGDLearner(new ParameterVector(modelOutput.Parameters().ToList()), learningRatePerSample, momentumPerSample, unitGain, GetAdditionalLearningOptions());
        }

        /// <summary>
        /// Adadelta is an extension of Adagrad that seeks to reduce its aggressive, monotonically decreasing learning rate. Instead of accumulating all past squared gradients, Adadelta restricts the window of accumulated past gradients to some fixed size w.
        /// </summary>
        /// <param name="modelOutput">The model output.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="rho">The rho.</param>
        /// <param name="epsilon">The epsilon.</param>
        /// <param name="regulizer">The regulizer.</param>
        /// <returns>Learner.</returns>
        private Learner AdaDelta(Function modelOutput, double learningRate = 1.0, double rho = 0.95, double epsilon = 1e-08, Regulizers regulizer = null)
        {
            CNTK.TrainingParameterScheduleDouble learningRatePerSample = new CNTK.TrainingParameterScheduleDouble(learningRate);
            return CNTKLib.AdaDeltaLearner(new ParameterVector(modelOutput.Parameters().ToList()), learningRatePerSample, rho, epsilon, GetAdditionalLearningOptions());
        }

        /// <summary>
        /// Adagrad is an algorithm for gradient-based optimization that does just this: It adapts the learning rate to the parameters, performing larger updates for infrequent and smaller updates for frequent parameters
        /// </summary>
        /// <param name="modelOutput">The model output.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="regulizer">The regulizer.</param>
        /// <returns>Learner.</returns>
        private Learner AdaGrad(Function modelOutput, double learningRate = 0.01, Regulizers regulizer = null)
        {
            CNTK.TrainingParameterScheduleDouble learningRatePerSample = new CNTK.TrainingParameterScheduleDouble(learningRate);
            return CNTKLib.AdaGradLearner(new ParameterVector(modelOutput.Parameters().ToList()), learningRatePerSample, false, GetAdditionalLearningOptions());
        }

        /// <summary>
        /// Adaptive Moment Estimation (Adam) is another method that computes adaptive learning rates for each parameter. In addition to storing an exponentially decaying average of past squared gradients vtvt like Adadelta and RMSprop, Adam also keeps an exponentially decaying average of past gradients mtmt, similar to momentum.
        /// </summary>
        /// <param name="modelOutput">The model output.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="momentum">The momentum.</param>
        /// <param name="varianceMomentum">The variance momentum.</param>
        /// <param name="unitGain">if set to <c>true</c> [unit gain].</param>
        /// <param name="epsilon">The epsilon.</param>
        /// <param name="regulizer">The regulizer.</param>
        /// <returns>Learner.</returns>
        private Learner Adam(Function modelOutput, double learningRate = 0.001, double momentum = 0.9, double varianceMomentum = 0.999, bool unitGain = true, double epsilon = 1e-08f, Regulizers regulizer = null)
        {
            CNTK.TrainingParameterScheduleDouble learningRatePerSample = new CNTK.TrainingParameterScheduleDouble(learningRate);
            CNTK.TrainingParameterScheduleDouble momentumRate = new CNTK.TrainingParameterScheduleDouble(momentum);
            CNTK.TrainingParameterScheduleDouble varianceMomentumRate = new CNTK.TrainingParameterScheduleDouble(varianceMomentum);
            return CNTKLib.AdamLearner(new ParameterVector(modelOutput.Parameters().ToList()), learningRatePerSample, momentumRate, unitGain, varianceMomentumRate, epsilon, false, GetAdditionalLearningOptions());
        }

        /// <summary>
        /// The Vt factor in the Adam update rule scales the gradient inversely proportionally to the ℓ2 norm of the past gradients (via the vt−1 term) and current gradient.
        /// </summary>
        /// <param name="modelOutput">The model output.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="momentum">The momentum.</param>
        /// <param name="varianceMomentum">The variance momentum.</param>
        /// <param name="unitGain">if set to <c>true</c> [unit gain].</param>
        /// <param name="epsilon">The epsilon.</param>
        /// <param name="regulizer">The regulizer.</param>
        /// <returns>Learner.</returns>
        private Learner Adamax(Function modelOutput, double learningRate = 0.002, double momentum = 0.9, double varianceMomentum = 0.999, bool unitGain = true, double epsilon = 1e-08, Regulizers regulizer = null)
        {
            CNTK.TrainingParameterScheduleDouble learningRatePerSample = new CNTK.TrainingParameterScheduleDouble(learningRate);
            CNTK.TrainingParameterScheduleDouble momentumRate = new CNTK.TrainingParameterScheduleDouble(momentum);
            CNTK.TrainingParameterScheduleDouble varianceMomentumRate = new CNTK.TrainingParameterScheduleDouble(varianceMomentum);
            return CNTKLib.AdamLearner(new ParameterVector(modelOutput.Parameters().ToList()), learningRatePerSample, momentumRate, unitGain, varianceMomentumRate, epsilon, true, GetAdditionalLearningOptions());
        }

        /// <summary>
        /// RMSprop is an unpublished, adaptive learning rate method proposed by Geoff Hinton. This optimizer is usually a good choice for recurrent neural networks.
        /// </summary>
        /// <param name="modelOutput">The model output.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="gamma">The gamma.</param>
        /// <param name="inc">The incremental value</param>
        /// <param name="dec">The decremental value.</param>
        /// <param name="min">The minimum.</param>
        /// <param name="max">The maximum.</param>
        /// <param name="regulizer">The regulizer.</param>
        /// <returns>Learner.</returns>
        private Learner RMSprop(Function modelOutput, double learningRate = 0.001, double gamma = 0.9, double inc = 2, double dec = 0.01, double min = 0.01, double max = 1, Regulizers regulizer = null)
        {
            CNTK.TrainingParameterScheduleDouble learningRatePerSample = new CNTK.TrainingParameterScheduleDouble(learningRate);
            return CNTKLib.RMSPropLearner(new ParameterVector(modelOutput.Parameters().ToList()), learningRatePerSample, gamma, inc, dec, max, min, false, GetAdditionalLearningOptions());
        }
    }
}
