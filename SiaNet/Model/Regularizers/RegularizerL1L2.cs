using CNTK;

namespace SiaNet.Model.Regularizers
{
    /// <inheritdoc />
    public class RegularizerL1L2 : RegularizerBase
    {
        /// <inheritdoc />
        public RegularizerL1L2(double l1 = 0.01, double l2 = 0.01) : base(false)
        {
            L1 = l1;
            L2 = l2;
        }

        /// <inheritdoc />
        public RegularizerL1L2(double l1, double l2, bool gradientClippingWithTruncation) : base(
            gradientClippingWithTruncation)
        {
            L1 = l1;
            L2 = l2;
        }

        /// <inheritdoc />
        public RegularizerL1L2(
            double l1,
            double l2,
            bool gradientClippingWithTruncation,
            double gradientClippingThresholdPerSample)
            : base(gradientClippingWithTruncation, gradientClippingThresholdPerSample)
        {
            L1 = l1;
            L2 = l2;
        }

        public double L1 { get; }
        public double L2 { get; }


        /// <inheritdoc />
        internal override AdditionalLearningOptions GetAdditionalLearningOptions()
        {
            var options = new AdditionalLearningOptions
            {
                l1RegularizationWeight = L1,
                l2RegularizationWeight = L2,
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