using SiaNet.Backend;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using System.Linq;

namespace SiaNet
{
    public partial class Sequential
    {
        public void Fit(DataIter train, uint epochs=1, uint batchSize=32, DataIter validation = null, bool shuffle = false)
        {
            var args = new SortedDictionary<string, NDArray>();
            string labelName = "label";
            var label = Symbol.Variable(labelName);
            
            List<uint> inputShape = new List<uint>();
            inputShape.Add(batchSize);
            inputShape.AddRange(InputShape.Data.Where(x =>(x > 0)));
            
            args["X"] = new NDArray(new Shape(inputShape.ToArray()));
            args[labelName] = new NDArray(new Shape(batchSize));
            
            CompiledModel.InferArgsMap(GlobalParam.Device, args, args);
            
            var defaultInitializer = new SiaDNN.Initializers.GlorotUniform();

            foreach (var arg in args)
            {
                if (ParamInitializers.ContainsKey(arg.Key))
                {
                    ParamInitializers[arg.Key].Operator(arg.Key, arg.Value);
                }
                else
                {
                    defaultInitializer.Operator(arg.Key, arg.Value);
                }
            }

            ModelOptimizer.SetParam("rescale_grad", 1.0 / batchSize);

            using (var exec = CompiledModel.SimpleBind(GlobalParam.Device, args))
            {
                var argNames = CompiledModel.ListArguments();

                // Start training
                var sw = new Stopwatch();
                for (var iter = 1; iter <= epochs; iter++)
                {
                    uint samples = 0;
                    train.BatchSize = batchSize;
                    train.Reset();
                    Metric.Reset();
                    TrainMetric.Reset();
                    sw.Restart();

                    while (train.Next())
                    {
                        samples += batchSize;
                        var dataBatch = train.GetDataBatch();
                        
                        // Set data and label
                        dataBatch.Data.CopyTo(args["X"]);
                        dataBatch.Label.CopyTo(args[labelName]);

                        // Compute gradients
                        exec.Forward(true);
                        exec.Backward();
                        // Update parameters
                        for (var i = 0; i < argNames.Count; ++i)
                        {
                            if (argNames[i] == "X" || argNames[i] == labelName)
                                continue;

                            ModelOptimizer.Update(i, exec.ArgmentArrays[i], exec.GradientArrays[i]);
                        }

                        TrainMetric.Update(dataBatch.Label, exec.Outputs[0]);
                    }
                    
                    sw.Stop();

                    if (validation != null)
                    {
                        validation.BatchSize = batchSize;
                        validation.Reset();
                        while (validation.Next())
                        {
                            var dataBatch = validation.GetDataBatch();
                            dataBatch.Data.CopyTo(args["X"]);
                            dataBatch.Label.CopyTo(args[labelName]);

                            // Forward pass is enough as no gradient is needed when evaluating
                            exec.Forward(false);
                            Metric.Update(dataBatch.Label, exec.Outputs[0]);
                        }
                    }


                    var duration = sw.ElapsedMilliseconds / 1000.0;
                    if (validation == null)
                    {
                        Logging.LG($"Epoch: {iter} {Convert.ToInt32(samples / duration)} samples/sec Train_Metric: {TrainMetric.Get()}");
                    }
                    else
                    {
                        Logging.LG($"Epoch: {iter} {Convert.ToInt32(samples / duration)} samples/sec, Train_Metric: {TrainMetric.Get()},  Val_Metric: {Metric.Get()}");
                    }
                }
            }

            MXNet.MXNotifyShutdown();
        }
    }
}
