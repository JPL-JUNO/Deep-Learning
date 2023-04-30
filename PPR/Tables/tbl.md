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
