/*
Copyright (c) 2015, ArrayFire
Copyright (c) 2015, Steven Burns (royalstream@hotmail.com)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of arrayfire_dotnet nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

using System;

using SiaNet.Backend.ArrayFire.Interop;

namespace SiaNet.Backend.ArrayFire
{
    public class ArrayFireException : Exception
    {
        public ArrayFireException(af_err message) : base(getError(message)) { }

        private static string getError(af_err err)
        {
            switch (err)
            {
                case af_err.AF_SUCCESS: return "Success";
                case af_err.AF_ERR_INTERNAL: return "Internal error";
                case af_err.AF_ERR_NO_MEM: return "Device out of memory";
                case af_err.AF_ERR_DRIVER: return "Driver not available or incompatible";
                case af_err.AF_ERR_RUNTIME: return "Runtime error ";
                case af_err.AF_ERR_INVALID_ARRAY: return "Invalid array";
                case af_err.AF_ERR_ARG: return "Invalid input argument";
                case af_err.AF_ERR_SIZE: return "Invalid input size";
                case af_err.AF_ERR_DIFF_TYPE: return "Input types are not the same";
                case af_err.AF_ERR_NOT_SUPPORTED: return "Function not supported";
                case af_err.AF_ERR_NOT_CONFIGURED: return "Function not configured to build";
                case af_err.AF_ERR_TYPE: return "Function does not support this data type";
                case af_err.AF_ERR_NO_DBL: return "Double precision not supported for this device";
                case af_err.AF_ERR_LOAD_LIB: return "Failed to load dynamic library";
                case af_err.AF_ERR_LOAD_SYM: return "Failed to load symbol";
                case af_err.AF_ERR_UNKNOWN: return "Unknown error";
                default: return err.ToString();
            }
        }
    }
}
