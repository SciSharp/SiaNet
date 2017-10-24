using CNTK;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet
{
    public class Optimizers
    {
        public static Learner Get(string optimizer, Function modelOutput)
        {
            switch (optimizer.Trim().ToLower())
            {
                case OptOptimizers.AdaDelta:
                    return AdaDelta(modelOutput);
                case OptOptimizers.AdaGrad:
                    return AdaGrad(modelOutput);
                case OptOptimizers.Adam:
                    return Adam(modelOutput);
                case OptOptimizers.Adamax:
                    return Adamax(modelOutput);
                case OptOptimizers.MomentumSGD:
                    return MomentumSGD(modelOutput);
                case OptOptimizers.RMSProp:
                    return RMSprop(modelOutput);
                case OptOptimizers.SGD:
                    return SGD(modelOutput);
                default:
                    throw new NotImplementedException(string.Format("{0} is not implemented", optimizer));
            }
        }

        public static Learner SGD(Function modelOutput, float learningRate = 0.01f)
        {
            CNTK.TrainingParameterScheduleDouble learningRatePerSample = new CNTK.TrainingParameterScheduleDouble(learningRate, 1);
            return CNTKLib.SGDLearner(new ParameterVector(modelOutput.Parameters().ToList()), learningRatePerSample);
        }

        public static Learner MomentumSGD(Function modelOutput, float learningRate = 0.01f, float momentum = 0, bool unitGain = false)
        {
            CNTK.TrainingParameterScheduleDouble learningRatePerSample = new CNTK.TrainingParameterScheduleDouble(learningRate, 1);
            CNTK.TrainingParameterScheduleDouble momentumPerSample = new CNTK.TrainingParameterScheduleDouble(momentum, 1);

            return CNTKLib.MomentumSGDLearner(new ParameterVector(modelOutput.Parameters().ToList()), learningRatePerSample, momentumPerSample, unitGain);
        }

        public static Learner AdaDelta(Function modelOutput, float learningRate = 1.0f, double rho = 0.95f, double epsilon = 1e-08f)
        {
            CNTK.TrainingParameterScheduleDouble learningRatePerSample = new CNTK.TrainingParameterScheduleDouble(learningRate, 1);
            return CNTKLib.AdaDeltaLearner(new ParameterVector(modelOutput.Parameters().ToList()), learningRatePerSample, rho, epsilon);
        }

        public static Learner AdaGrad(Function modelOutput, float learningRate = 0.01f)
        {
            CNTK.TrainingParameterScheduleDouble learningRatePerSample = new CNTK.TrainingParameterScheduleDouble(learningRate, 1);
            return CNTKLib.AdaGradLearner(new ParameterVector(modelOutput.Parameters().ToList()), learningRatePerSample);
        }

        public static Learner Adam(Function modelOutput, float learningRate = 0.001f, float momentum=0.9f, float varianceMomentum=0.999f, bool unitGain=false, double epsilon = 1e-08f)
        {
            CNTK.TrainingParameterScheduleDouble learningRatePerSample = new CNTK.TrainingParameterScheduleDouble(learningRate, 1);
            CNTK.TrainingParameterScheduleDouble momentumRate = new CNTK.TrainingParameterScheduleDouble(momentum, 1);
            CNTK.TrainingParameterScheduleDouble varianceMomentumRate = new CNTK.TrainingParameterScheduleDouble(varianceMomentum, 1);
            return CNTKLib.AdamLearner(new ParameterVector(modelOutput.Parameters().ToList()), learningRatePerSample, momentumRate, unitGain, varianceMomentumRate, epsilon, false);
        }

        public static Learner Adamax(Function modelOutput, float learningRate = 0.002f, float momentum = 0.9f, float varianceMomentum = 0.999f, bool unitGain = false, double epsilon = 1e-08f)
        {
            CNTK.TrainingParameterScheduleDouble learningRatePerSample = new CNTK.TrainingParameterScheduleDouble(learningRate, 1);
            CNTK.TrainingParameterScheduleDouble momentumRate = new CNTK.TrainingParameterScheduleDouble(momentum, 1);
            CNTK.TrainingParameterScheduleDouble varianceMomentumRate = new CNTK.TrainingParameterScheduleDouble(varianceMomentum, 1);
            return CNTKLib.AdamLearner(new ParameterVector(modelOutput.Parameters().ToList()), learningRatePerSample, momentumRate, unitGain, varianceMomentumRate, epsilon, true);
        }

        public static Learner RMSprop(Function modelOutput, float learningRate = 0.001f, float gamma = 0.9f, float inc = 2, float dec = 0.01f, double min = 0.01, double max = 10)
        {
            CNTK.TrainingParameterScheduleDouble learningRatePerSample = new CNTK.TrainingParameterScheduleDouble(learningRate, 1);
            return CNTKLib.RMSPropLearner(new ParameterVector(modelOutput.Parameters().ToList()), learningRatePerSample, gamma, inc, dec, min, max);
        }
    }
}
