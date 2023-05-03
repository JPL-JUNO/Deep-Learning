# Tables fot Notes

## Table: Tensor data Types

| Data Type                       | dtype                         | CPU tensor         | GPU tensor              |
| ------------------------------- | ----------------------------- | ------------------ | ----------------------- |
| 32-bit floating point (default) | torch.float32 or torch.float  | torch.FloatTensor  | torch.cuda.FloatTensor  |
| 64-bit floating point           | torch.float64 or torch.double | torch.DoubleTensor | torch.cuda.DoubleTensor |
| 16-bit floating point           | torch.float16 or torch.half   | torch.HalfTensor   | torch.cuda.HalfTensor   |
| 8-bit integer (unsigned)        | torch.uint8                   | torch.ByteTensor   | torch.cuda.ByteTensor   |
| 8-bit integer (signed)          | torch.int8                    | torch.CharTensor   | torch.cuda.CharTensor   |
| 16-bit integer (signed)         | torch.int16 or torch.short    | torch.ShortTensor  | torch.cuda.ShortTensor  |
| 32-bit integer (signed)         | torch.int32 or torch.int      | torch.IntTensor    | torch.cuda.IntTensor    |
| 64-bit integer (signed)         | torch.int64 or torch.long     | torch. LongTensor  | torch.cuda.LongTensor   |
| Boolean                         | torch.bool                    | torch.BoolTensor   | torch.cuda.BoolTensor   |

## Table: Random sampling functions

| Function                                                                                                                          | Description                                                                              |
| --------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| torch.rand(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)                                   | Selects random values from a uniform distribution on the interval [0 to 1]               |
| torch.randn(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)                                  | Selects random values from a standard normal distribution with zero mean unit variance   |
| torch.normal(mean, std, *, generator=None, out=None)                                                                              | Selects random numbers from a normal distribution with a specified mean and variance     |
| torch.randint(low=0, high, size, *, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) | Selects random integers generated uniformly between specified low and high values        |
| torch.randperm(n, out=None, dtype=torch.int64, layout=torch.strided, device=None, requires_grad=False)                            | Creates a random permutation of integers from 0 to n–1                                   |
| torch.bernoulli(input, *, generator=None, out=None)                                                                               | Draws binary random numbers (0 or 1) from a Bernoulli distribution                       |
| torch.multinomial(input, num_samples, replacement=False, *, generator=None, out=None)                                             | Selects a random number from a list according to weights from a multinomial distribution |

## Table:Indexing, slicing, combining, and splitting operations

| Function              | Description                                                                                                                                                   |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| torch.cat()           | Concatenates the given sequence of tensors in the given dimension.                                                                                            |
| torch.chunk()         | Splits a tensor into a specific number of chunks. Each chunk is a view of the input tensor.                                                                   |
| torch.gather()        | Gathers values along an axis specified by the dimension.                                                                                                      |
| torch.index_select()  | Returns a new tensor that indexes the input tensor along a dimension using the entries in the index, which is a LongTensor.                                   |
| torch.masked_select() | Returns a new 1D tensor that indexes the input tensor according to the Boolean mask, which is a BoolTensor.                                                   |
| torch.narrow()        | Returns a tensor that is a narrow version of the input tensor.                                                                                                |
| torch.nonzero()       | Returns the indices of nonzero elements.                                                                                                                      |
| torch.reshape()       | Returns a tensor with the same data and number of elements as the input tensor, but a different shape. Use view() instead to ensure the tensor is not copied. |
| torch.split()         | Splits the tensor into chunks. Each chunk is a view or subdivision of the original tensor.                                                                    |
| torch.squeeze()       | Returns a tensor with all the dimensions of the input tensor of size 1 removed.                                                                               |
| torch.stack()         | Concatenates a sequence of tensors along a new dimension.                                                                                                     |
| torch.t()             | Expects the input to be a 2D tensor and transposes dimensions 0 and 1.                                                                                        |
| torch.take()          | Returns a tensor at specified indices when slicing is not continuous.                                                                                         |
| torch.transpose()     | Transposes only the specified dimensions.                                                                                                                     |
| torch.unbind()        | Removes a tensor dimension by returning a tuple of the removed dimension.                                                                                     |
| torch.unsqueeze()     | Returns a new tensor with a dimension of size 1 inserted at the specified position.                                                                           |
| torch.where()         | Returns a tensor of selected elements from either one of two tensors, depending on the specified condition.                                                   |

## Table: Pointwise operations

| Operation type           | Sample functions                                                                                           |
| ------------------------ | ---------------------------------------------------------------------------------------------------------- |
| Basic math               | add(), div(), mul(), neg(), reciprocal(), true_divide()                                                    |
| Truncation               | ceil(), clamp(), floor(), floor_divide(), fmod(), frac(), lerp(), remainder(), round(), sigmoid(), trunc() |
| Complex numbers          | abs(), angle(), conj(), imag(), real()                                                                     |
| Trigonometry             | acos(), asin(), atan(), cos(), cosh(), deg2rad(), rad2deg(), sin(), sinh(), tan(), tanh()                  |
| Exponents and logarithms | exp(), expm1(), log(), log10(), log1p(), log2(), logaddexp(), pow(), rsqrt(), sqrt(), square()             |
| Logical                  | logical_and(), logical_not(), logical_or(), logical_xor()                                                  |
| Cumulative math          | addcdiv(), addcmul()                                                                                       |
| Bitwise operators        | bitwise_not(), bitwise_and(), bitwise_or(), bitwise_xor()                                                  |
| Error functions          | erf(), erfc(), erfinv()                                                                                    |
| Gamma functions          | digamma(), lgamma(), mvlgamma(), polygamma()                                                               |

## Table: Reduction operations

| Reduction operations                                                      | Function Description                                                                                  |
| ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| torch.argmax(input, dim, keepdim=False, out=None)                         | Returns the index(es) of the maximum value across all elements, or just a dimension if it’s specified |
| torch.argmin(input, dim, keepdim=False, out=None)                         | Returns the index(es) of the minimum value across all elements, or just a dimension if it’s specified |
| torch.dist(input, dim, keepdim=False, out=None)                           | Computes the p-norm of two tensors                                                                    |
| torch.logsumexp(input, dim, keepdim=False, out=None)                      | Computes the log of summed exponentials of each row of the input tensor in the given dimension        |
| torch.mean(input, dim, keepdim=False, out=None)                           | Computes the mean or average across all elements, or just a dimension if it’s specified               |
| torch.median(input, dim, keepdim=False, out=None)                         | Computes the median or middle value across all elements, or just a dimension if it’s specified        |
| torch.mode(input, dim, keepdim=False, out=None)                           | Computes the mode or most frequent value across all elements, or just a dimension if it’s specified   |
| torch.norm(input, p='fro', dim=None, keepdim=False, out=None, dtype=None) | Computes the matrix or vector norm across all elements, or just a dimension if it’s specified         |
| torch.prod(input, dim, keepdim=False, dtype=None)                         | Computes the product of all elements, or of each row of the input tensor if it’s specified            |
| torch.std(input, dim, keepdim=False, out=None)                            | Computes the standard deviation across all elements, or just a dimension if it’s specified            |
| torch.std_mean(input, unbiased=True)                                      | Computes the standard deviation and mean across all elements, or just a dimension if it’s specified   |
| torch.sum(input, dim, keepdim=False, out=None)                            | Computes the sum of all elements, or just a dimension if it’s specified                               |
| torch.unique(input, dim, keepdim=False, out=None)                         | Removes duplicates across the entire tensor, or just a dimension if it’s specified                    |
| torch.unique_consecutive(input, dim, keepdim=False, out=None)             | Similar to torch.unique() but only removes consecutive duplicates                                     |
| torch.var(input, dim, keepdim=False, out=None)                            | Computes the variance across all elements, or just a dimension if it’s specified                      |
| torch.var_mean(input, dim, keepdim=False, out=None)                       | Computes the mean and variance across all elements, or just a dimension if it’s specified             |

## Table Comparison operations

| Operation type                                                  | Sample functions                                                         |
| --------------------------------------------------------------- | ------------------------------------------------------------------------ |
| Compare a tensor to other tensors                               | eq(), ge(), gt(), le(), lt(), ne() or ==, >, >=, <, <=, !=, respectively |
| Test tensor status or conditions                                | isclose(), isfinite(), isinf(), isnan()                                  |
| Return a single Boolean for the entire tensor                   | allclose(), equal()                                                      |
| Find value(s) over the entire tensor or along a given dimension | argsort(), kthvalue(), max(), min(), sort(), topk()                      |

## Table: Linear algebra operations

| Function                 | Description                                                                                  |
| ------------------------ | -------------------------------------------------------------------------------------------- |
| torch.matmul()           | Computes a matrix product of two tensors; supportsbroadcasting                               |
| torch.chain_matmul()     | Computes a matrix product of N tensors                                                       |
| torch.mm()               | Computes a matrix product of two tensors (if broadcasting is required, use matmul())         |
| torch.addmm()            | Computes a matrix product of two tensors and adds it to the input                            |
| torch.bmm()              | Computes a batch of matrix products                                                          |
| torch.addbmm()           | Computes a batch of matrix products and adds it to the input                                 |
| torch.baddbmm()          | Computes a batch of matrix products and adds it to the input batch                           |
| torch.mv()               | Computes the product of the matrix and vector                                                |
| torch.addmv()            | Computes the product of the matrix and vector and adds it to the input                       |
| torch.matrix_power       | Returns a tensor raised to the power of n (for square tensors)                               |
| torch.eig()              | Finds the eigenvalues and eigenvectors of a real square tensor                               |
| torch.inverse()          | Computes the inverse of a square tensor                                                      |
| torch.det()              | Computes the determinant of a matrix or batch of matrices                                    |
| torch.logdet()           | Computes the log determinant of a matrix or batch of matrices                                |
| torch.dot()              | Computes the inner product of two tensors                                                    |
| torch.addr()             | Computes the outer product of two tensors and adds it to the input                           |
| torch.solve()            | Returns the solution to a system of linear equations                                         |
| torch.svd()              | Performs a single-value decomposition                                                        |
| torch.pca_lowrank()      | Performs a linear principle component analysis                                               |
| torch.cholesky()         | Computes a Cholesky decomposition                                                            |
| torch.cholesky_inverse() | Computes the inverse of a symmetric positive definite matrix and returns the Cholesky factor |
| torch.cholesky_solve()   | Solves a system of linear equations using theCholesky factor                                 |

## Table: Spectral and other math operations

| Operation type                                                            | Sample functions                                                                                                            |
| ------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Fast, inverse, and short-time Fourier transforms                          | fft(), ifft(), stft()                                                                                                       |
| Real-to-complex FFT and complex-to-real inverse FFT(IFFT)                 | rfft(), irfft()                                                                                                             |
| Windowing algorithms                                                      | bartlett_window(), blackman_window(), hamming_window(), hann_window()                                                       |
| Histogram and bin counts                                                  | histc(), bincount()                                                                                                         |
| Cumulative operations                                                     | cummax(), cummin(), cumprod(), cumsum(), trace() (sum of the diagonal), einsum() (sum of products using Einstein summation) |
| Normalization functions                                                   | cdist(), renorm()                                                                                                           |
| Cross product, dot product, and Cartesian product                         | cross(), tensordot(), cartesian_prod()                                                                                      |
| Functions that create a diagonal tensor with elements of the input tensor | diag(), diag_embed(), diag_flat(), diagonal()                                                                               |
| Einstein summation                                                        | einsum()                                                                                                                    |
| Matrix reduction and restructuring functions                              | flatten(), flip(), rot90(), repeat_interleave(), meshgrid(), roll(), combinations()                                         |
| Functions that return the lower or upper triangles and their indices      | tril(), tril_indices, triu(), triu_indices()                                                                                |

