using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TP2;

namespace TP3
{
    class Program
    {

        static List<Vector<double>> ParseTSV(string path)
        {
            List<Vector<double>> trainingInput = new List<Vector<double>>();
            using (var reader = new StreamReader(path))
            {
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    var values = line.Split((char[])null, StringSplitOptions.RemoveEmptyEntries).Select(val => Double.Parse(val));
                    trainingInput.Add(Vector<double>.Build.Dense(values.ToArray()));
                }
            }
            return trainingInput;
        }
        static List<Vector<double>> Normalize(IEnumerable<Vector<double>> values, double rangeBottom, double rangeTop)
        {
            if (values.Count() == 0) return values.ToList();
            var max = values.Max(v => v.Max());
            var min = values.Min(v => v.Min());
            return values.Select(v => Normalize(v,max,min,rangeBottom,rangeTop)).ToList();
        }
        static Vector<double> Normalize(Vector<double> vector, double max, double min, double rangeBottom, double rangeTop)
        {
            return vector.Map(value => ((value - min) / (max - min)) * (rangeTop - rangeBottom) + rangeBottom);
        }
        static void PrintOutput(Perceptron p, IEnumerable<Vector<double>> input, IEnumerable<Vector<double>> output, string activation)
        {
            if (input.Count() == 0) return;
            var max = output.Max(v => v.Max());
            var min = output.Min(v => v.Min());
            for (int i = 0; i < input.Count(); i++)
            {
                var value = p.Map(input.ElementAt(i));
                var outputVal = output.ElementAt(i);
                if (activation == "tanh")
                    value = Normalize(value, 1, -1, min, max);
                else if (activation == "sigmoid")
                    value = Normalize(value, 1, 0, min, max);
                Console.WriteLine($"" +
                    $"Expected:{string.Join(',', output.ElementAt(i))}    " +
                    $"Value:{string.Join(',', value)}   " +
                    $"Error:{string.Join(',', 0.5 * (output.ElementAt(i) - value) * (output.ElementAt(i) - value))}");
            }
        }
        public static double Sigmoid(double value)
        {
            return 1.0 / (1.0 + Math.Exp(-value));
        }
        static List<(Vector<double>[], Vector<double>[])> GetAllTestGroups(IEnumerable<Vector<double>> training, int testSize)
        {
            var groups = new List<(Vector<double>[], Vector<double>[])>();
            if (testSize == 0)
            {
                groups.Add((training.ToArray(), new Vector<double>[0]));
            }
            else
            {
                for (int i = 0; i < training.Count() / testSize; i++)
                {
                    groups.Add((training.Take(testSize * i).Concat(training.Skip(testSize * (i+1))).ToArray(), training.Skip(testSize * i).Take(testSize).ToArray()));
                }
            }
            return groups;
        }


        /// <summary>
        /// Simple and Multi-layer Perceptrons
        /// </summary>
        /// <param name="config">Path to the configuration file</param>
        static void Main(string config)
        {
            Configuration configuration = Configuration.FromYamlFile(config);

            //Obtengo los conjuntos de entrenamiento y prueba.
            List<Vector<double>> trainingInput = new List<Vector<double>>();
            List<Vector<double>> trainingOutput = new List<Vector<double>>();
            var inputs = new List<(Vector<double>[] training, Vector<double>[] testing)>();
            var outputs = new List<(Vector<double>[] training, Vector<double>[] testing)>();
            try
            {
                trainingInput = ParseTSV(configuration.TrainingInput);
                trainingOutput = ParseTSV(configuration.TrainingOutput);
                if (configuration.CrossValidation)
                {
                    inputs = GetAllTestGroups(trainingInput, (int)Math.Round(configuration.TestSize.Value * trainingInput.Count));
                    outputs = GetAllTestGroups(trainingOutput, (int)Math.Round(configuration.TestSize.Value * trainingInput.Count));
                }
                else
                {
                    inputs.Add(
                        (trainingInput.GetRange((int)Math.Round(trainingInput.Count * (1 - configuration.TestSize.Value)),(int)Math.Round(trainingInput.Count * configuration.TestSize.Value)).ToArray(),
                        trainingInput.GetRange(0, (int)Math.Round(trainingInput.Count * (1 - configuration.TestSize.Value))).ToArray()));
                    outputs.Add(
                        (trainingOutput.GetRange((int)Math.Round(trainingOutput.Count * (1 - configuration.TestSize.Value)), (int)Math.Round(trainingOutput.Count * configuration.TestSize.Value)).ToArray(),
                        trainingOutput.GetRange(0, (int)Math.Round(trainingOutput.Count * (1 - configuration.TestSize.Value))).ToArray()));
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("Ha ocurrido un error al leer los archivos de entrenamiento.");
                Console.WriteLine(ex.ToString());
                return;
            }

            //Obtengo la funcion de activacion.
            Func<double, double> activationFunction;
            Func<double, double> activationFunctionD;
            var normalizedOutputs = new List<(Vector<double>[] training, Vector<double>[] testing)>();
            switch (configuration.Activation)
            {
                case "step":
                    activationFunction = val => Math.Sign(val);
                    activationFunctionD = val => 1;
                    normalizedOutputs = outputs;
                    break;
                case "linear":
                    activationFunction = val => val;
                    activationFunctionD = val => 1;
                    normalizedOutputs = outputs;
                    break;
                case "tanh":
                    activationFunction = Math.Tanh;
                    activationFunctionD = val => (1 - Math.Tanh(val)*Math.Tanh(val));
                    normalizedOutputs = outputs.Select(o => (Normalize(o.training, -1, 1).ToArray(), Normalize(o.testing, -1, 1).ToArray())).ToList();
                    break;
                case "sigmoid":
                    activationFunction = Sigmoid;
                    activationFunctionD = val => Sigmoid(val)*(1 - Sigmoid(val));
                    normalizedOutputs = outputs.Select(o => (Normalize(o.training, 0, 1).ToArray(), Normalize(o.testing, 0, 1).ToArray())).ToList();
                    break;
                default: Console.WriteLine("Ingrese la función de activación."); return;
            }
            Perceptron perceptron;
            switch (configuration.Type)
            {
                case "simple":
                    perceptron = new SimplePerceptron(
                        trainingInput[0].Count,
                        configuration.LearningRate,
                        activationFunction,
                        activationFunctionD);
                    break;
                case "multilayer":
                    var activations = new Func<double, double>[configuration.Layers.Length];
                    Array.Fill(activations, activationFunction);
                    var activationsD = new Func<double, double>[configuration.Layers.Length];
                    Array.Fill(activationsD, activationFunctionD);
                    perceptron = new MultiLayerPerceptron(
                        configuration.Layers,
                        configuration.LearningRate,
                        activations,
                        activationsD);
                    break;
                default: Console.WriteLine("Ingrese el tipo de perceptrón.");return;
            }
            //Validacion cruzada para obtener el mejor conjunto de prueba
            double minError = -1;
            (Vector<double>[] training, Vector<double>[] testing) optimumInput = (null, null);
            (Vector<double>[] training, Vector<double>[] testing) optimumOutput = (null, null);
            for (int i = 0; i < inputs.Count; i++)
            {
                var input = inputs[i];
                var output = normalizedOutputs[i];
                perceptron.Learn(input.training.ToArray(), output.training.ToArray(), input.testing.ToArray(), 
                    output.testing.ToArray(), configuration.Batch.Value, configuration.MinError.Value, configuration.Epochs);
                var error = perceptron.CalculateError(input.testing.ToArray(), output.testing.ToArray());
                if (error < minError || minError == -1)
                {
                    minError = error;
                    optimumInput = input;
                    optimumOutput = outputs[i];
                }

            }
            Console.WriteLine("Training Set: ");
            PrintOutput(perceptron, optimumInput.training, optimumOutput.training, configuration.Activation);
            Console.WriteLine();
            Console.WriteLine("Testing Set: ");
            PrintOutput(perceptron, optimumInput.testing, optimumOutput.testing, configuration.Activation);
            Console.WriteLine();
            Console.WriteLine($"Total Testing Error: {minError}");
        }

    }
}
