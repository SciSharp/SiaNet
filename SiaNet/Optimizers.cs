using CNTK;
using SiaNet.Common;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet
{
    public class Optimizers
    {
        public static Learner Get(string optimizer, Function modelOutput, Regulizers regulizer = null)
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

        public static Learner SGD(Function modelOutput, float learningRate = 0.01f, Regulizers regulizer = null)
        {
            CNTK.TrainingParameterScheduleDouble learningRatePerSample = new CNTK.TrainingParameterScheduleDouble(learningRate, 1);
            return CNTKLib.SGDLearner(new ParameterVector(modelOutput.Parameters().ToList()), learningRatePerSample, GetAdditionalLearningOptions(regulizer));
        }

        public static Learner MomentumSGD(Function modelOutput, float learningRate = 0.01f, float momentum = 0, bool unitGain = true, Regulizers regulizer = null)
        {
            CNTK.TrainingParameterScheduleDouble learningRatePerSample = new CNTK.TrainingParameterScheduleDouble(learningRate, 1);
            CNTK.TrainingParameterScheduleDouble momentumPerSample = new CNTK.TrainingParameterScheduleDouble(momentum, 1);

            return CNTKLib.MomentumSGDLearner(new ParameterVector(modelOutput.Parameters().ToList()), learningRatePerSample, momentumPerSample, unitGain, GetAdditionalLearningOptions(regulizer));
        }

        public static Learner AdaDelta(Function modelOutput, float learningRate = 1.0f, double rho = 0.95f, double epsilon = 1e-08f, Regulizers regulizer = null)
        {
            CNTK.TrainingParameterScheduleDouble learningRatePerSample = new CNTK.TrainingParameterScheduleDouble(learningRate, 1);
            return CNTKLib.AdaDeltaLearner(new ParameterVector(modelOutput.Parameters().ToList()), learningRatePerSample, rho, epsilon, GetAdditionalLearningOptions(regulizer));
        }

        public static Learner AdaGrad(Function modelOutput, float learningRate = 0.01f, Regulizers regulizer = null)
        {
            CNTK.TrainingParameterScheduleDouble learningRatePerSample = new CNTK.TrainingParameterScheduleDouble(learningRate, 1);
            return CNTKLib.AdaGradLearner(new ParameterVector(modelOutput.Parameters().ToList()), learningRatePerSample,false, GetAdditionalLearningOptions(regulizer));
        }

        public static Learner Adam(Function modelOutput, float learningRate = 0.001f, float momentum=0.9f, float varianceMomentum=0.999f, bool unitGain=true, double epsilon = 1e-08f, Regulizers regulizer = null)
        {
            CNTK.TrainingParameterScheduleDouble learningRatePerSample = new CNTK.TrainingParameterScheduleDouble(learningRate, 1);
            CNTK.TrainingParameterScheduleDouble momentumRate = new CNTK.TrainingParameterScheduleDouble(momentum, 1);
            CNTK.TrainingParameterScheduleDouble varianceMomentumRate = new CNTK.TrainingParameterScheduleDouble(varianceMomentum, 1);
            return CNTKLib.AdamLearner(new ParameterVector(modelOutput.Parameters().ToList()), learningRatePerSample, momentumRate, unitGain, varianceMomentumRate, epsilon, false, GetAdditionalLearningOptions(regulizer));
        }

        public static Learner Adamax(Function modelOutput, float learningRate = 0.002f, float momentum = 0.9f, float varianceMomentum = 0.999f, bool unitGain = true, double epsilon = 1e-08f, Regulizers regulizer = null)
        {
            CNTK.TrainingParameterScheduleDouble learningRatePerSample = new CNTK.TrainingParameterScheduleDouble(learningRate, 1);
            CNTK.TrainingParameterScheduleDouble momentumRate = new CNTK.TrainingParameterScheduleDouble(momentum, 1);
            CNTK.TrainingParameterScheduleDouble varianceMomentumRate = new CNTK.TrainingParameterScheduleDouble(varianceMomentum, 1);
            return CNTKLib.AdamLearner(new ParameterVector(modelOutput.Parameters().ToList()), learningRatePerSample, momentumRate, unitGain, varianceMomentumRate, epsilon, true, GetAdditionalLearningOptions(regulizer));
        }

        public static Learner RMSprop(Function modelOutput, float learningRate = 0.001f, float gamma = 0.9f, float inc = 2, float dec = 0.01f, double min = 0.01, double max = 1, Regulizers regulizer = null)
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
