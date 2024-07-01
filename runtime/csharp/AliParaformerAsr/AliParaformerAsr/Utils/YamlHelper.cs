// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Text.Json;
using YamlDotNet.Serialization;

namespace AliParaformerAsr.Utils
{
    /// <summary>
    /// YamlHelper
    /// Copyright (c)  2023 by manyeyes
    /// </summary>
    internal class YamlHelper 
    {
        public static T ReadYaml<T>(string yamlFilePath) where T:new()
        {
            if (!File.Exists(yamlFilePath))
            {
                // 如果允许返回默认对象，则新建一个默认对象，否则应该是抛出异常
                // If allowing to return a default object, create a new default object; otherwise, throw an exception

                return new T();
                // throw new Exception($"not find yaml config file: {yamlFilePath}");
            }

            StreamReader yamlReader = File.OpenText(yamlFilePath);
            Deserializer yamlDeserializer = new Deserializer();
            T info = yamlDeserializer.Deserialize<T>(yamlReader);
            yamlReader.Close();
            return info;
        }
    }
}
