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
        static void PrintOutput(Perceptron p, List<Vector<double>> input, List<Vector<double>> output, string activation)
        {
            if (input.Count == 0) return;
            var max = output.Max(v => v.Max());
            var min = output.Min(v => v.Min());
            for (int i = 0; i < input.Count; i++)
            {
                var value = p.Map(input[i]);
                var outputVal = output[i];
                if (activation == "tanh")
                    value = Normalize(value, 1, -1, min, max);
                else if (activation == "sigmoid")
                    value = Normalize(value, 1, 0, min, max);
                Console.WriteLine($"" +
                    $"Expected:{string.Join(',', output[i])}    " +
                    $"Value:{string.Join(',', value)}   " +
                    $"Error:{string.Join(',', 0.5 * (output[i] - value) * (output[i] - value))}");
            }
        }
        public static double Sigmoid(double value)
        {
            return 1.0 / (1.0 + Math.Exp(-value));
        }
        /// <summary>
        /// Simple and Multi-layer Perceptrons
        /// </summary>
        /// <param name="config">Path to the configuration file</param>
        static void Main(string config)
        {
            Configuration configuration = Configuration.FromYamlFile(config);

            //Obtengo los conjuntos de entrenamiento.
            List<Vector<double>> trainingInput = new List<Vector<double>>();
            List<Vector<double>> testInput = new List<Vector<double>>();
            List<Vector<double>> trainingOutput = new List<Vector<double>>();
            List<Vector<double>> testOutput = new List<Vector<double>>();
            try
            {
                trainingInput = ParseTSV(configuration.TrainingInput);
                testInput = trainingInput.GetRange(
                    (int)Math.Round(trainingInput.Count * (1 - configuration.TestSize.Value)),
                    (int)Math.Round(trainingInput.Count * configuration.TestSize.Value));
                trainingInput = trainingInput.GetRange(0, (int)Math.Round(trainingInput.Count * (1 - configuration.TestSize.Value)));
                trainingOutput = ParseTSV(configuration.TrainingOutput);
                testOutput = trainingOutput.GetRange(
                    (int)Math.Round(trainingOutput.Count * (1 - configuration.TestSize.Value)),
                    (int)Math.Round(trainingOutput.Count * configuration.TestSize.Value));
                trainingOutput = trainingOutput.GetRange(0, (int)Math.Round(trainingOutput.Count * (1 - configuration.TestSize.Value)));
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
            List<Vector<double>> normalizedTrainOutput = new List<Vector<double>>();
            List<Vector<double>> normalizedTestOutput = new List<Vector<double>>();
            switch (configuration.Activation)
            {
                case "step":
                    activationFunction = val => Math.Sign(val);
                    activationFunctionD = val => 1;
                    normalizedTrainOutput = trainingOutput;
                    normalizedTestOutput = testOutput;
                    break;
                case "linear":
                    activationFunction = val => val;
                    activationFunctionD = val => 1;
                    normalizedTrainOutput = trainingOutput;
                    normalizedTestOutput = testOutput;
                    break;
                case "tanh":
                    activationFunction = Math.Tanh;
                    activationFunctionD = val => (1 - Math.Tanh(val)*Math.Tanh(val));
                    normalizedTrainOutput = Normalize(trainingOutput, -1, 1);
                    normalizedTestOutput = Normalize(testOutput, -1, 1);
                    break;
                case "sigmoid":
                    activationFunction = Sigmoid;
                    activationFunctionD = val => Sigmoid(val)*(1 - Sigmoid(val));
                    normalizedTrainOutput = Normalize(trainingOutput, 0, 1);
                    normalizedTestOutput = Normalize(testOutput, 0, 1);
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
                    ((SimplePerceptron)perceptron).Learn(configuration.Epochs, trainingInput.ToArray(), normalizedTrainOutput.Select(v => v.At(0)).ToArray());
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
                    ((MultiLayerPerceptron)perceptron).Learn(trainingInput.ToArray(), normalizedTrainOutput.ToArray(), configuration.Batch.Value, configuration.MinError.Value, configuration.Epochs);
                    break;
                default: Console.WriteLine("Ingrese el tipo de perceptrón.");return;
            }
            Console.WriteLine("Training Set: ");
            PrintOutput(perceptron, trainingInput, trainingOutput, configuration.Activation);
            Console.WriteLine();
            Console.WriteLine("Testing Set: ");
            PrintOutput(perceptron, testInput, testOutput, configuration.Activation);
        }

    }
}
