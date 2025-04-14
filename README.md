# Neural-Networks-from-Scratch

Этот проект представляет собой реализацию нейронной сети с нуля на C++. Он предназначен для обучения и тестирования
нейросетей на любом наборе входных и выходных данных.

## Структура проекта

Проект имеет следующую структуру:

- net: Класс, представляющий нейронную сеть. Он содержит слои (layer), а также хранит функцию
  потерь (loss_function) во время обучения.
- layer: Класс, представляющий слой нейронной сети. Он содержит линейную функцию и функцию активации (
  activation_function).
- activation_function: Класс, реализующий различные функции активации, такие как Sigmoid, ReLU, Tanh, Linear и Softmax.
- loss_function: Класс, реализующий функции потерь, используемые для обучения нейронной сети, такие как MSE, Manhattan и
  CrossEntropy.
- definitions: Файл, содержащий определения типов данных, используемых в проекте.

## Использование

Пример использования проекта:

```cpp
#include "Network/net.h"

DataSet dataset(GetMnistData(kDataSize));

const Sizes k_layer_sizes = {dataset.num_input_pixels, 32, dataset.num_output_pixels};
const AFNames k_af_names = {AFName::ReLU, AFName::Softmax};
Net net(k_layer_sizes, k_af_names, input_path);

const LFName kDefaultLFName = LFName::MSE;
const PrintInfo kDefaultPI = PrintInfo::PrintInfo;
MnistTesting::Train(net, dataset, kIterCount, kIlr, kDecay, kLFName, output_path);

Net::Info info = net.Train(dataset.train, kLFName, kDefaultError, kDefaultMaxIter,
                     kDefaultInitLR, kDefaultDecay, kDefaultBatchSize,
                     kDefaultPI);
net.SaveParams(output_path);
std::cout << "RESULT:\n"
          << "iterations: " << info.iterations_count << "\terror rate: " << info.error_rate
          << "\n";

DataType accuracy = CalcAccuracy(net, dataset);
std::cout << "Accuracy of Neural Network: " << accuracy << "\n\n";

```

В этом примере:

1. Загружается набор данных MNIST размером 60000 изображений.
2. Создается нейронная сеть net с заданными размерами слоев и функциями активации (ReLU и Softmax).
3. Определяются параметры обучения, такие как функция потерь (CrossEntropy), количество итераций и скорость обучения.
4. Выполняется обучение нейронной сети с помощью метода Train.
5. Вычисляется точность обученной нейронной сети на тестовом наборе данных.

## Зависимости

Проект не имеет внешних зависимостей и может быть скомпилирован с использованием стандартного компилятора C++.

