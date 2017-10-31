using SiaNet.Common;
using System;
using System.Collections.Generic;
using System.Dynamic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SiaNet.Model.Layers
{
    public class BatchNorm : LayerConfig
    {
        public BatchNorm()
        {
            base.Name = "BatchNorm";
            base.Params = new ExpandoObject();
        }

        public BatchNorm(float epsilon = 0.001f, string betaInitializer = OptInitializers.Zeros, string gammaInitializers = OptInitializers.Ones,
                                       string runningMeanInitializer = OptInitializers.Zeros, string runningStdInvInitializer = OptInitializers.Ones, bool spatial = true,
                                       float normalizationTimeConstant = 4096f, float blendTimeConst = 0.0f)
            : this()
        {
            Shape = null;
            Epsilon = epsilon;
            BetaInitializer = betaInitializer;
            GammaInitializer = gammaInitializers;
            RunningMeanInitializer = runningMeanInitializer;
            RunningStdInvInitializer = runningStdInvInitializer;
            Spatial = spatial;
            NormalizationTimeConstant = normalizationTimeConstant;
            BlendTimeConst = blendTimeConst;
        }

        public BatchNorm(int shape, float epsilon = 0.001f, string betaInitializer = OptInitializers.Zeros, string gammaInitializers = OptInitializers.Ones,
                                       string runningMeanInitializer = OptInitializers.Zeros, string runningStdInvInitializer = OptInitializers.Ones, bool spatial = true,
                                       float normalizationTimeConstant = 4096f, float blendTimeConst = 0.0f)
            : this(epsilon, betaInitializer, gammaInitializers, runningMeanInitializer, runningStdInvInitializer, spatial, normalizationTimeConstant, blendTimeConst)
        {
            Shape = shape;
        }

        [Newtonsoft.Json.JsonIgnore]
        public int? Shape
        {
            get
            {
                return base.Params.Shape;
            }

            set
            {
                base.Params.Shape = value;
            }
        }

        [Newtonsoft.Json.JsonIgnore]
        public float Epsilon
        {
            get
            {
                return base.Params.Epsilon;
            }

            set
            {
                base.Params.Epsilon = value;
            }
        }

        [Newtonsoft.Json.JsonIgnore]
        public string BetaInitializer
        {
            get
            {
                return base.Params.BetaInitializer;
            }

            set
            {
                base.Params.BetaInitializer = value;
            }
        }

        [Newtonsoft.Json.JsonIgnore]
        public string GammaInitializer
        {
            get
            {
                return base.Params.GammaInitializer;
            }

            set
            {
                base.Params.GammaInitializer = value;
            }
        }

        [Newtonsoft.Json.JsonIgnore]
        public string RunningMeanInitializer
        {
            get
            {
                return base.Params.RunningMeanInitializer;
            }

            set
            {
                base.Params.RunningMeanInitializer = value;
            }
        }

        [Newtonsoft.Json.JsonIgnore]
        public string RunningStdInvInitializer
        {
            get
            {
                return base.Params.RunningStdInvInitializer;
            }

            set
            {
                base.Params.RunningStdInvInitializer = value;
            }
        }

        [Newtonsoft.Json.JsonIgnore]
        public bool Spatial
        {
            get
            {
                return base.Params.Spatial;
            }

            set
            {
                base.Params.Spatial = value;
            }
        }

        [Newtonsoft.Json.JsonIgnore]
        public float NormalizationTimeConstant
        {
            get
            {
                return base.Params.NormalizationTimeConstant;
            }

            set
            {
                base.Params.NormalizationTimeConstant = value;
            }
        }

        [Newtonsoft.Json.JsonIgnore]
        public float BlendTimeConst
        {
            get
            {
                return base.Params.BlendTimeConst;
            }

            set
            {
                base.Params.BlendTimeConst = value;
            }
        }
    }
}
