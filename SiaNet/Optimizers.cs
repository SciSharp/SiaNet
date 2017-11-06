using CNTK;
using SiaNet.Common;
using System;
namespace SiaNet
{
    using System.Linq;

    /// <summary>
    /// An optimizer is one of the three arguments required for compiling a model. The choice of optimization algorithm for your deep learning model can mean the difference between good results in minutes, hours, and days.
    /// <see cref="OptOptimizers"/>
    /// </summary>
    internal class Optimizers
    {
        internal static Learner Get(string optimizer, Function modelOutput, Regulizers regulizer = null)
        {
            switch (optimizer.Trim().ToLower())
            {
                case OptOptimizers.AdaDelta:
                    return AdaDelta(modelOutput, regulizer: regulizer);
                case OptOptimizers.AdaGrad:
                    return AdaGrad(modelOutput, regulizer: regulizer);
                case OptOptimizers.Adam:
                    return Adam(modelOutput, regulizer: regulizer);
                case OptOptimizers.Adamax:
                    return Adamax(modelOutput, regulizer: regulizer);
                case OptOptimizers.MomentumSGD:
                    return MomentumSGD(modelOutput, regulizer: regulizer);
                case OptOptimizers.RMSProp:
                    return RMSprop(modelOutput, regulizer: regulizer);
                case OptOptimizers.SGD:
                    return SGD(modelOutput, regulizer: regulizer);
                default:
                    throw new NotImplementedException(string.Format("{0} is not implemented", optimizer));
            }
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
        private static Learner SGD(Function modelOutput, float learningRate = 0.01f, Regulizers regulizer = null)
        {
            CNTK.TrainingParameterScheduleDouble learningRatePerSample = new CNTK.TrainingParameterScheduleDouble(learningRate, 1);
            return CNTKLib.SGDLearner(new ParameterVector(modelOutput.Parameters().ToList()), learningRatePerSample, GetAdditionalLearningOptions(regulizer));
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
        private static Learner MomentumSGD(Function modelOutput, float learningRate = 0.01f, float momentum = 0, bool unitGain = true, Regulizers regulizer = null)
        {
            CNTK.TrainingParameterScheduleDouble learningRatePerSample = new CNTK.TrainingParameterScheduleDouble(learningRate, 1);
            CNTK.TrainingParameterScheduleDouble momentumPerSample = new CNTK.TrainingParameterScheduleDouble(momentum, 1);

            return CNTKLib.MomentumSGDLearner(new ParameterVector(modelOutput.Parameters().ToList()), learningRatePerSample, momentumPerSample, unitGain, GetAdditionalLearningOptions(regulizer));
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
        private static Learner AdaDelta(Function modelOutput, float learningRate = 1.0f, double rho = 0.95f, double epsilon = 1e-08f, Regulizers regulizer = null)
        {
            CNTK.TrainingParameterScheduleDouble learningRatePerSample = new CNTK.TrainingParameterScheduleDouble(learningRate, 1);
            return CNTKLib.AdaDeltaLearner(new ParameterVector(modelOutput.Parameters().ToList()), learningRatePerSample, rho, epsilon, GetAdditionalLearningOptions(regulizer));
        }

        /// <summary>
        /// Adagrad is an algorithm for gradient-based optimization that does just this: It adapts the learning rate to the parameters, performing larger updates for infrequent and smaller updates for frequent parameters
        /// </summary>
        /// <param name="modelOutput">The model output.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="regulizer">The regulizer.</param>
        /// <returns>Learner.</returns>
        private static Learner AdaGrad(Function modelOutput, float learningRate = 0.01f, Regulizers regulizer = null)
        {
            CNTK.TrainingParameterScheduleDouble learningRatePerSample = new CNTK.TrainingParameterScheduleDouble(learningRate, 1);
            return CNTKLib.AdaGradLearner(new ParameterVector(modelOutput.Parameters().ToList()), learningRatePerSample,false, GetAdditionalLearningOptions(regulizer));
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
        private static Learner Adam(Function modelOutput, float learningRate = 0.001f, float momentum=0.9f, float varianceMomentum=0.999f, bool unitGain=true, double epsilon = 1e-08f, Regulizers regulizer = null)
        {
            CNTK.TrainingParameterScheduleDouble learningRatePerSample = new CNTK.TrainingParameterScheduleDouble(learningRate, 1);
            CNTK.TrainingParameterScheduleDouble momentumRate = new CNTK.TrainingParameterScheduleDouble(momentum, 1);
            CNTK.TrainingParameterScheduleDouble varianceMomentumRate = new CNTK.TrainingParameterScheduleDouble(varianceMomentum, 1);
            return CNTKLib.AdamLearner(new ParameterVector(modelOutput.Parameters().ToList()), learningRatePerSample, momentumRate, unitGain, varianceMomentumRate, epsilon, false, GetAdditionalLearningOptions(regulizer));
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
        private static Learner Adamax(Function modelOutput, float learningRate = 0.002f, float momentum = 0.9f, float varianceMomentum = 0.999f, bool unitGain = true, double epsilon = 1e-08f, Regulizers regulizer = null)
        {
            CNTK.TrainingParameterScheduleDouble learningRatePerSample = new CNTK.TrainingParameterScheduleDouble(learningRate, 1);
            CNTK.TrainingParameterScheduleDouble momentumRate = new CNTK.TrainingParameterScheduleDouble(momentum, 1);
            CNTK.TrainingParameterScheduleDouble varianceMomentumRate = new CNTK.TrainingParameterScheduleDouble(varianceMomentum, 1);
            return CNTKLib.AdamLearner(new ParameterVector(modelOutput.Parameters().ToList()), learningRatePerSample, momentumRate, unitGain, varianceMomentumRate, epsilon, true, GetAdditionalLearningOptions(regulizer));
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
        private static Learner RMSprop(Function modelOutput, float learningRate = 0.001f, float gamma = 0.9f, float inc = 2, float dec = 0.01f, double min = 0.01, double max = 1, Regulizers regulizer = null)
        {
            CNTK.TrainingParameterScheduleDouble learningRatePerSample = new CNTK.TrainingParameterScheduleDouble(learningRate, 1);
            return CNTKLib.RMSPropLearner(new ParameterVector(modelOutput.Parameters().ToList()), learningRatePerSample, gamma, inc, dec, max, min, false, GetAdditionalLearningOptions(regulizer));
        }

        private static AdditionalLearningOptions GetAdditionalLearningOptions(Regulizers regulizer)
        {
            AdditionalLearningOptions options = new AdditionalLearningOptions();
            if (regulizer != null)
            {
                if (regulizer.IsL1)
                    options.l1RegularizationWeight = regulizer.L1;
                if (regulizer.IsL2)
                    options.l1RegularizationWeight = regulizer.L2;
            }

            return options;
        }
    }
}
