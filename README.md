﻿# Simple and Multilayer Perceptron Engine

## Build
Requires .Net 3.1 SDK

```console
dotnet build -c Release
```

The project can also be imported in Visual Studio 2019

## Run

```console
cd TP3\bin\Release\netcoreapp3.1\
.\TP3.exe [options]
```
Information about valid arguments can be found with the --help command

```console
.\TP3.exe --help

TP3
  Simple and Multi-layer Perceptrons

Usage:
  TP3 [options]

Options:
  --config <config>  Path to the configuration file
  --version          Show version information
  -?, -h, --help     Show help and usage information
```

## Configuration

```console
.\TP3.exe --config config.yaml
```

The configuration file must have the following format:

```console
training_input: <input file path>
input_lines: <amount of lines for single input> (default=1)
training_output: <desired output file path>
output_lines: <amount of lines for output input> (default=1)
type: <simple|multilayer>
activation: <step|linear|tanh|sigmoid>
learning_rate: <learning rate>
adaptive_learning_rate: <false|true> (default=false)
epochs: <epochs limit>
test_size: <test size percentage> (default=0.0)
batch: <batch_size> (default=1)
layers: <layers size array> (only if type=multilayer; ex: [2,2,1])
min_error: <min error limit> (default=0)
cross_validation: <false|true> (default=false)
cross_validation_k: <amount of training groups for cross validation> (default=10)
```



