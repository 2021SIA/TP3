using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TP2;
using static TP3.PerceptronUtils;

namespace TP3
{
    class Program
    {

        static List<Vector<double>> ParseTSV(string path, int recordLines)
        {
            List<Vector<double>> trainingInput = new List<Vector<double>>();
            using (var reader = new StreamReader(path))
            {
                while (!reader.EndOfStream)
                {
                    var line = "";
                    for(var i = 0; i < recordLines; i++)
                        line += " " + reader.ReadLine();
                    var values = line.Split((char[])null, StringSplitOptions.RemoveEmptyEntries).Select(val => Double.Parse(val));
                    trainingInput.Add(Vector<double>.Build.Dense(values.ToArray()));
                }
            }
            return trainingInput;
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
            Console.WriteLine();
            //Si usa una funcion de activacion de escalon, es un metodo de clasificacion (en dos clases)
            if(activation == "step")
            {
                ClassifierMetrics metrics = GetClassifierMetrics(p, input, output);
                Console.WriteLine($"Accuracy: {metrics.Accuracy}");
                Console.WriteLine($"Precision: {metrics.Precision}");
                Console.WriteLine($"Recall: {metrics.Recall}");
                Console.WriteLine($"F1-Score: {metrics.F1Score}");
            }
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
                trainingInput = ParseTSV(configuration.TrainingInput,configuration.InputLines);
                trainingOutput = ParseTSV(configuration.TrainingOutput,configuration.OutputLines);

                //Si se usa validacion cruzada, obtengo los k conjuntos de entrenamiento con su respectivo conjunto de prueba.
                if (configuration.CrossValidation && configuration.TestSize != 0.0)
                {
                    inputs = GetAllTestGroups(trainingInput, (int)Math.Round(configuration.TestSize * trainingInput.Count));
                    outputs = GetAllTestGroups(trainingOutput, (int)Math.Round(configuration.TestSize * trainingInput.Count));
                }
                //Si no se usa validacion cruzada, solo tomo como conjunto de prueba los ultimos TestSize del conjunto de entrenamiento.
                else
                {
                    inputs.Add(
                        (trainingInput.GetRange(0, (int)Math.Round(trainingInput.Count * (1 - configuration.TestSize))).ToArray(),
                        trainingInput.GetRange((int)Math.Round(trainingInput.Count * (1 - configuration.TestSize)),(int)Math.Round(trainingInput.Count * configuration.TestSize)).ToArray()));
                    outputs.Add(
                        (trainingOutput.GetRange(0, (int)Math.Round(trainingOutput.Count * (1 - configuration.TestSize))).ToArray(),
                        trainingOutput.GetRange((int)Math.Round(trainingOutput.Count * (1 - configuration.TestSize)), (int)Math.Round(trainingOutput.Count * configuration.TestSize)).ToArray()));
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
                    //Normalizo el output deseado entre los valores -1 y 1.
                    normalizedOutputs = outputs.Select(o => (Normalize(o.training, -1, 1).ToArray(), Normalize(o.testing, -1, 1).ToArray())).ToList();
                    break;
                case "sigmoid":
                    activationFunction = Sigmoid;
                    activationFunctionD = val => Sigmoid(val)*(1 - Sigmoid(val));
                    //Normalizo el output deseado entre los valores 0 y 1.
                    normalizedOutputs = outputs.Select(o => (Normalize(o.training, 0, 1).ToArray(), Normalize(o.testing, 0, 1).ToArray())).ToList();
                    break;
                default: Console.WriteLine("Ingrese la función de activación."); return;
            }
            //Inicializo el perceptron.
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

            (Vector<double>[] training, Vector<double>[] testing) optimumInput = (null, null);
            (Vector<double>[] training, Vector<double>[] testing) optimumOutput = (null, null);
            (Vector<double>[] training, Vector<double>[] testing) optimumNormalizedOutput = (null, null);
            //Validacion cruzada para obtener el mejor conjunto de prueba.
            if (configuration.CrossValidation && configuration.TestSize != 0)
            {
                double minError = -1;
                for (int i = 0; i < inputs.Count; i++)
                {
                    var input = inputs[i];
                    var output = normalizedOutputs[i];
                    perceptron.Learn(input.training.ToArray(), output.training.ToArray(), input.testing.ToArray(),
                        output.testing.ToArray(), configuration.Batch, configuration.MinError, configuration.Epochs);
                    var error = perceptron.CalculateError(input.testing.ToArray(), output.testing.ToArray());
                    if (error < minError || minError == -1)
                    {
                        minError = error;
                        optimumInput = input;
                        optimumOutput = outputs[i];
                        optimumNormalizedOutput = normalizedOutputs[i];
                    }

                }
            }
            else
            {
                optimumInput = inputs[0];
                optimumOutput = outputs[0];
                optimumNormalizedOutput = normalizedOutputs[0];
            }
            //Entreno al perceptron con el input y output optimo encontrado por el metodo de validacion cruzada.
            perceptron.Learn(optimumInput.training.ToArray(), optimumNormalizedOutput.training.ToArray(), optimumInput.testing.ToArray(),
                optimumNormalizedOutput.testing.ToArray(), configuration.Batch, configuration.MinError, configuration.Epochs);
            
            Console.WriteLine("Training Set: ");
            PrintOutput(perceptron, optimumInput.training, optimumOutput.training, configuration.Activation);
            var totalError = perceptron.CalculateError(optimumInput.training.ToArray(), optimumNormalizedOutput.training.ToArray());
            Console.WriteLine($"Total Training Error: {totalError}");
            Console.WriteLine();
            if(optimumInput.testing.Length > 0)
            {
                Console.WriteLine("Testing Set: ");
                PrintOutput(perceptron, optimumInput.testing, optimumOutput.testing, configuration.Activation);
                totalError = perceptron.CalculateError(optimumInput.testing.ToArray(), optimumNormalizedOutput.testing.ToArray());
                Console.WriteLine($"Total Testing Error: {totalError}");
                Console.WriteLine();
            }
        }

    }
}
