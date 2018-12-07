// ReSharper disable once CheckNamespace
namespace SiaNet.Backend
{

    public class FactorScheduler : LRScheduler
    {

        #region Fields

        private int _Count;

        private readonly int _Step;

        private readonly float _Factor;

        private readonly float _StopFactorLearningRate;

        #endregion

        #region Constructors

        public FactorScheduler(int step, float factor = 1, float stopFactorLearningRate = 1e-8f)
        {
            this._Step = step;
            this._Factor = factor;
            this._StopFactorLearningRate = stopFactorLearningRate;
        }

        #endregion

        #region Methods

        public override float GetLearningRate(uint numUpdate)
        {
            while (numUpdate > (uint)(this._Count + this._Step))
            {
                this._Count += this._Step;

                this.BaseLearningRate *= this._Factor;
                if (this.BaseLearningRate < this._StopFactorLearningRate)
                {
                    this.BaseLearningRate = this._StopFactorLearningRate;
                    Logging.LG($"Update[{numUpdate}]: now learning rate arrived at {this.BaseLearningRate}, will not change in the future");
                }
                else
                {
                    Logging.LG($"Update[{numUpdate}]: Change learning rate to {this.BaseLearningRate}");
                }
            }
            return this.BaseLearningRate;
        }

        #endregion

    }

}