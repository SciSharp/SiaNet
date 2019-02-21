using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp.CUDA;
using TensorSharp.CUDA.DeviceCode;

namespace TensorSharp.Cuda.KernelOps
{
    [OpsClass]
    public class MatrixOps
    {
        private readonly MatrixKernels matrixKernels = new MatrixKernels();

        public MatrixOps()
        {

        }

        [RegisterOpStorageType("diag", typeof(CudaStorage))]
        public Tensor Diag(Tensor src) { return matrixKernels.Diag(src); }
    }
}
