namespace SiaNet.EventArgs
{
    public class TrainingEndEventArgs : System.EventArgs
    {
        public TrainingEndEventArgs(
            double loss,
            double validationLoss,
            double metric,
            double validationMetric)
        {
            Loss = loss;
            ValidationLoss = validationLoss;
            Metric = metric;
            ValidationMetric = validationMetric;
        }

        public double Loss { get; }
        public double Metric { get; }
        public double ValidationLoss { get; }
        public double ValidationMetric { get; }
    }
}