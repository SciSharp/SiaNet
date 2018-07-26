namespace SiaNet.Model.Trainer.DQN
{
    public interface IEnvironment
    {
        EnvironmentState Reset();
        EnvironmentState Step(int action);
    }
}
