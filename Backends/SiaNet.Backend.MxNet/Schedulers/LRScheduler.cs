// ReSharper disable once CheckNamespace
namespace SiaNet.Backend.MxNetLib
{

    public abstract class LRScheduler
    {

        #region Constructors

        protected LRScheduler(float baseLearningRate = 0.01f)
        {
            this.BaseLearningRate = baseLearningRate;
        }

        #endregion

        #region Properties

        protected float BaseLearningRate
        {
            get;
            set;
        }

        #endregion

        #region Methods

        public void SetLearningRate(float learningRate)
        {
            this.BaseLearningRate = learningRate;
        }

        public abstract float GetLearningRate(uint numUpdate);

        #endregion

    }

}
