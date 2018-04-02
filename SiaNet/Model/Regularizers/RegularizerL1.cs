using CNTK;

namespace SiaNet.Model.Regularizers
{
    /// <inheritdoc />
    public class RegularizerL1 : RegularizerBase
    {
        /// <inheritdoc />
        public RegularizerL1(double l1 = 0.01) : base(false)
        {
            L1 = l1;
        }

        /// <inheritdoc />
        public RegularizerL1(double l1, bool gradientClippingWithTruncation) : base(gradientClippingWithTruncation)
        {
            L1 = l1;
        }

        /// <inheritdoc />
        public RegularizerL1(double l1, bool gradientClippingWithTruncation, double gradientClippingThresholdPerSample)
            : base(gradientClippingWithTruncation, gradientClippingThresholdPerSample)
        {
            L1 = l1;
        }

        public double L1 { get; }

        /// <inheritdoc />
        internal override AdditionalLearningOptions GetAdditionalLearningOptions()
        {
            var options = new AdditionalLearningOptions
            {
                l1RegularizationWeight = L1,
                gradientClippingWithTruncation = GradientClippingWithTruncation
            };

            if (DoesHaveGradientClippingThresholdPerSample)
            {
                options.gradientClippingThresholdPerSample = GradientClippingThresholdPerSample;
            }

            return options;
        }
    }
}