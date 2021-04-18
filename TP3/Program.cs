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

        /// <summary>
        /// Simple and Multi-layer Perceptrons
        /// </summary>
        /// <param name="config">Path to the configuration file</param>
        static void Main(string config)
        {
            Configuration configuration = Configuration.FromYamlFile(config);

            //Obtengo los conjuntos de entrenamiento.
            List<Vector<double>> trainingInput = new List<Vector<double>>();
            List<Vector<double>> trainingOutput = new List<Vector<double>>();
            try
            {
                trainingInput = ParseTSV(configuration.TrainingInput);
                trainingOutput = ParseTSV(configuration.TrainingOutput);
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
            switch (configuration.Activation)
            {
                case "step":
                    activationFunction = val => Math.Sign(val);
                    activationFunctionD = val => 1;
                    break;
                case "linear":
                    activationFunction = val => val;
                    activationFunctionD = val => 1;
                    break;
                case "tanh":
                    activationFunction = Math.Tanh;
                    activationFunctionD = val => (Math.Cosh(val)*Math.Cosh(val) - Math.Sinh(val)*Math.Sinh(val)) / (Math.Cosh(val)*Math.Cosh(val));
                    break;
                default: Console.WriteLine("Ingrese la función de activación."); return;
            }

            switch (configuration.Type)
            {
                case "simple":
                    SimplePerceptron simplePerceptron = new SimplePerceptron(
                        trainingInput[0].Count,
                        configuration.LearningRate,
                        activationFunction,
                        activationFunctionD);
                    simplePerceptron.Learn(configuration.Epochs, trainingInput.ToArray(), trainingOutput.Select(v => v.At(0)).ToArray());
                    break;
                case "multilayer":
                    var activations = new Func<double, double>[configuration.Layers.Length];
                    Array.Fill(activations, activationFunction);
                    var activationsD = new Func<double, double>[configuration.Layers.Length];
                    Array.Fill(activationsD, activationFunctionD);
                    MultiLayerPerceptron multilayerPerceptron = new MultiLayerPerceptron(
                        configuration.Layers,
                        configuration.LearningRate,
                        activations,
                        activationsD);
                    multilayerPerceptron.Learn(trainingInput.ToArray(), trainingOutput.ToArray(), configuration.Batch.Value, configuration.MinError.Value, configuration.Epochs);
                    break;
                default: Console.WriteLine("Ingrese el tipo de perceptrón.");return;
            }
        }
    }
}
