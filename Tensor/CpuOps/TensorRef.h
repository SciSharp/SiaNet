#pragma once

enum DType
{
	Float32 = 0,
	Float16 = 1,
	Float64 = 2,
	Int32 = 3,
	UInt8 = 4,
};

struct TensorRef
{
	void* buffer;
	__int64* sizes;
	__int64* strides;
	int dimCount;
	DType elementType;


	__int64 ElementCount()
	{
		if (dimCount == 0)
			return 0;

		__int64 total = 1;
		for (int i = 0; i < dimCount; ++i)
			total *= sizes[i];
		return total;
	}


	bool IsContiguous()
	{
		__int64 z = 1;
		for (int d = dimCount - 1; d >= 0; d--)
		{
			if (sizes[d] != 1)
			{
				if (strides[d] == z)
					z *= sizes[d];
				else
					return false;
			}
		}
		return true;
	}

};
