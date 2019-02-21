// This file was automatically generated using the AutoGenTool project
// If possible, edit the tool instead of editing this file directly

using System;
using System.Text;
using System.Numerics;
using System.Security;
using System.Runtime.InteropServices;

namespace SiaNet.Backend.ArrayFire.Interop
{
	public enum af_err
	{
		///
		/// The function returned successfully
		///
		AF_SUCCESS            =   0,

		// 100-199 Errors in environment

		///
		/// The system or device ran out of memory
		///
		AF_ERR_NO_MEM         = 101,

		///
		/// There was an error in the device driver
		///
		AF_ERR_DRIVER         = 102,

		///
		/// There was an error with the runtime environment
		///
		AF_ERR_RUNTIME        = 103,

		// 200-299 Errors in input parameters

		///
		/// The input array is not a valid af_array object
		///
		AF_ERR_INVALID_ARRAY  = 201,

		///
		/// One of the function arguments is incorrect
		///
		AF_ERR_ARG            = 202,

		///
		/// The size is incorrect
		///
		AF_ERR_SIZE           = 203,

		///
		/// The type is not suppported by this function
		///
		AF_ERR_TYPE           = 204,

		///
		/// The type of the input arrays are not compatible
		///
		AF_ERR_DIFF_TYPE      = 205,

		///
		/// Function does not support GFOR / batch mode
		///
		AF_ERR_BATCH          = 207,


		// #if AF_API_VERSION >= 33
		///
		/// Input does not belong to the current device.
		///
		AF_ERR_DEVICE         = 208,
		// #endif

		// 300-399 Errors for missing software features

		///
		/// The option is not supported
		///
		AF_ERR_NOT_SUPPORTED  = 301,

		///
		/// This build of ArrayFire does not support this feature
		///
		AF_ERR_NOT_CONFIGURED = 302,

		// #if AF_API_VERSION >= 32
		///
		/// This build of ArrayFire is not compiled with "nonfree" algorithms
		///
		AF_ERR_NONFREE        = 303,
		// #endif

		// 400-499 Errors for missing hardware features

		///
		/// This device does not support double
		///
		AF_ERR_NO_DBL         = 401,

		///
		/// This build of ArrayFire was not built with graphics or this device does
		/// not support graphics
		///
		AF_ERR_NO_GFX         = 402,

		// 500-599 Errors specific to heterogenous API

		// #if AF_API_VERSION >= 32
		///
		/// There was an error when loading the libraries
		///
		AF_ERR_LOAD_LIB       = 501,
		// #endif

		// #if AF_API_VERSION >= 32
		///
		/// There was an error when loading the symbols
		///
		AF_ERR_LOAD_SYM       = 502,
		// #endif

		// #if AF_API_VERSION >= 32
		///
		/// There was a mismatch between the input array and the active backend
		///
		AF_ERR_ARR_BKND_MISMATCH    = 503,
		// #endif

		// 900-999 Errors from upstream libraries and runtimes

		///
		/// There was an internal error either in ArrayFire or in a project
		/// upstream
		///
		AF_ERR_INTERNAL       = 998,

		///
		/// Unknown Error
		///
		AF_ERR_UNKNOWN        = 999
	}

	public enum af_dtype
	{
		f32,    ///< 32-bit floating point values
		c32,    ///< 32-bit complex floating point values
		f64,    ///< 64-bit complex floating point values
		c64,    ///< 64-bit complex floating point values
		b8 ,    ///< 8-bit boolean values
		s32,    ///< 32-bit signed integral values
		u32,    ///< 32-bit unsigned integral values
		u8 ,    ///< 8-bit unsigned integral values
		s64,    ///< 64-bit signed integral values
		u64,    ///< 64-bit unsigned integral values
		// #if AF_API_VERSION >= 32
		s16,    ///< 16-bit signed integral values
		// #endif
		// #if AF_API_VERSION >= 32
		u16,    ///< 16-bit unsigned integral values
		// #endif
	}

	public enum af_source
	{
		afDevice,   ///< Device pointer
		afHost,     ///< Host pointer
	}

	public enum af_interp_type
	{
		AF_INTERP_NEAREST,  ///< Nearest Interpolation
		AF_INTERP_LINEAR,   ///< Linear Interpolation
		AF_INTERP_BILINEAR, ///< Bilinear Interpolation
		AF_INTERP_CUBIC,    ///< Cubic Interpolation
		AF_INTERP_LOWER     ///< Floor Indexed
	}

	public enum af_border_type
	{
		///
		/// Out of bound values are 0
		///
		AF_PAD_ZERO = 0,

		///
		/// Out of bound values are symmetric over the edge
		///
		AF_PAD_SYM
	}

	public enum af_connectivity
	{
		///
		/// Connectivity includes neighbors, North, East, South and West of current pixel
		///
		AF_CONNECTIVITY_4 = 4,

		///
		/// Connectivity includes 4-connectivity neigbors and also those on Northeast, Northwest, Southeast and Southwest
		///
		AF_CONNECTIVITY_8 = 8
	}

	public enum af_conv_mode
	{
		///
		/// Output of the convolution is the same size as input
		///
		AF_CONV_DEFAULT,

		///
		/// Output of the convolution is signal_len + filter_len - 1
		///
		AF_CONV_EXPAND,
	}

	public enum af_conv_domain
	{
		AF_CONV_AUTO,    ///< ArrayFire automatically picks the right convolution algorithm
		AF_CONV_SPATIAL, ///< Perform convolution in spatial domain
		AF_CONV_FREQ,    ///< Perform convolution in frequency domain
	}

	public enum af_match_type
	{
		AF_SAD = 0,   ///< Match based on Sum of Absolute Differences (SAD)
		AF_ZSAD,      ///< Match based on Zero mean SAD
		AF_LSAD,      ///< Match based on Locally scaled SAD
		AF_SSD,       ///< Match based on Sum of Squared Differences (SSD)
		AF_ZSSD,      ///< Match based on Zero mean SSD
		AF_LSSD,      ///< Match based on Locally scaled SSD
		AF_NCC,       ///< Match based on Normalized Cross Correlation (NCC)
		AF_ZNCC,      ///< Match based on Zero mean NCC
		AF_SHD        ///< Match based on Sum of Hamming Distances (SHD)
	}

	public enum af_ycc_std
	{
		AF_YCC_601 = 601,  ///< ITU-R BT.601 (formerly CCIR 601) standard
		AF_YCC_709 = 709,  ///< ITU-R BT.709 standard
		AF_YCC_2020 = 2020  ///< ITU-R BT.2020 standard
	}

	public enum af_cspace_t
	{
		AF_GRAY = 0, ///< Grayscale
		AF_RGB,      ///< 3-channel RGB
		AF_HSV,      ///< 3-channel HSV
		// #if AF_API_VERSION >= 31
		AF_YCbCr     ///< 3-channel YCbCr
		// #endif
	}

	public enum af_mat_prop
	{
		AF_MAT_NONE       = 0,    ///< Default
		AF_MAT_TRANS      = 1,    ///< Data needs to be transposed
		AF_MAT_CTRANS     = 2,    ///< Data needs to be conjugate tansposed
		AF_MAT_CONJ       = 4,    ///< Data needs to be conjugate
		AF_MAT_UPPER      = 32,   ///< Matrix is upper triangular
		AF_MAT_LOWER      = 64,   ///< Matrix is lower triangular
		AF_MAT_DIAG_UNIT  = 128,  ///< Matrix diagonal contains unitary values
		AF_MAT_SYM        = 512,  ///< Matrix is symmetric
		AF_MAT_POSDEF     = 1024, ///< Matrix is positive definite
		AF_MAT_ORTHOG     = 2048, ///< Matrix is orthogonal
		AF_MAT_TRI_DIAG   = 4096, ///< Matrix is tri diagonal
		AF_MAT_BLOCK_DIAG = 8192  ///< Matrix is block diagonal
	}

	public enum af_norm_type
	{
		AF_NORM_VECTOR_1,      ///< treats the input as a vector and returns the sum of absolute values
		AF_NORM_VECTOR_INF,    ///< treats the input as a vector and returns the max of absolute values
		AF_NORM_VECTOR_2,      ///< treats the input as a vector and returns euclidean norm
		AF_NORM_VECTOR_P,      ///< treats the input as a vector and returns the p-norm
		AF_NORM_MATRIX_1,      ///< return the max of column sums
		AF_NORM_MATRIX_INF,    ///< return the max of row sums
		AF_NORM_MATRIX_2,      ///< returns the max singular value). Currently NOT SUPPORTED
		AF_NORM_MATRIX_L_PQ,   ///< returns Lpq-norm

		AF_NORM_EUCLID = AF_NORM_VECTOR_2, ///< The default. Same as AF_NORM_VECTOR_2
	}

	public enum af_colormap
	{
		AF_COLORMAP_DEFAULT = 0,    ///< Default grayscale map
		AF_COLORMAP_SPECTRUM= 1,    ///< Spectrum map
		AF_COLORMAP_COLORS  = 2,    ///< Colors
		AF_COLORMAP_RED     = 3,    ///< Red hue map
		AF_COLORMAP_MOOD    = 4,    ///< Mood map
		AF_COLORMAP_HEAT    = 5,    ///< Heat map
		AF_COLORMAP_BLUE    = 6     ///< Blue hue map
	}

	public enum af_image_format
	{
		AF_FIF_BMP          = 0,    ///< FreeImage Enum for Bitmap File
		AF_FIF_ICO          = 1,    ///< FreeImage Enum for Windows Icon File
		AF_FIF_JPEG         = 2,    ///< FreeImage Enum for JPEG File
		AF_FIF_JNG          = 3,    ///< FreeImage Enum for JPEG Network Graphics File
		AF_FIF_PNG          = 13,   ///< FreeImage Enum for Portable Network Graphics File
		AF_FIF_PPM          = 14,   ///< FreeImage Enum for Portable Pixelmap (ASCII) File
		AF_FIF_PPMRAW       = 15,   ///< FreeImage Enum for Portable Pixelmap (Binary) File
		AF_FIF_TIFF         = 18,   ///< FreeImage Enum for Tagged Image File Format File
		AF_FIF_PSD          = 20,   ///< FreeImage Enum for Adobe Photoshop File
		AF_FIF_HDR          = 26,   ///< FreeImage Enum for High Dynamic Range File
		AF_FIF_EXR          = 29,   ///< FreeImage Enum for ILM OpenEXR File
		AF_FIF_JP2          = 31,   ///< FreeImage Enum for JPEG-2000 File
		AF_FIF_RAW          = 34    ///< FreeImage Enum for RAW Camera Image File
	}

	public enum af_homography_type
	{
		AF_HOMOGRAPHY_RANSAC = 0,   ///< Computes homography using RANSAC
		AF_HOMOGRAPHY_LMEDS  = 1    ///< Computes homography using Least Median of Squares
	}

	public enum af_backend
	{
		AF_BACKEND_DEFAULT = 0,  ///< Default backend order: OpenCL -> CUDA -> CPU
		AF_BACKEND_CPU     = 1,  ///< CPU a.k.a sequential algorithms
		AF_BACKEND_CUDA    = 2,  ///< CUDA Compute Backend
		AF_BACKEND_OPENCL  = 4,  ///< OpenCL Compute Backend
	}

	public enum af_someenum_t
	{
		AF_ID = 0
	}

	public enum af_marker_type
	{
		AF_MARKER_NONE         = 0,
		AF_MARKER_POINT        = 1,
		AF_MARKER_CIRCLE       = 2,
		AF_MARKER_SQUARE       = 3,
		AF_MARKER_TRIANGLE     = 4,
		AF_MARKER_CROSS        = 5,
		AF_MARKER_PLUS         = 6,
		AF_MARKER_STAR         = 7
	}
}
