using System.IO;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace TP2
{
    public class Configuration
    {
        public string TrainingInput { get; set; }
        public string TrainingOutput { get; set; }
        public string Type { get; set; }
        public string Activation { get; set; }
        public double LearningRate { get; set; }
        public int Epochs { get; set; }
        public double? TestSize { get; set; } = 0.0;
        public int? Batch { get; set; } = 1;
        public int[] Layers { get; set; } = null;
        public double? MinError { get; set; } = 0.0;
        public static Configuration FromYamlFile(string path)
        {
            var deserializer = new DeserializerBuilder()
                .WithNamingConvention(UnderscoredNamingConvention.Instance)
                .Build();
            return deserializer.Deserialize<Configuration>(File.OpenText(path));
        }
        public Configuration() { }
    }
}
