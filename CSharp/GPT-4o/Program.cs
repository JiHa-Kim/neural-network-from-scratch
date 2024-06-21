Console.WriteLine("Running GPT-4o");
Main();
// Done in 7 rounds, followed prompt very well

double Sigmoid(double x)
{
    return 1.0 / (1.0 + Math.Exp(-x));
}

double SigmoidDerivative(double x)
{
    return x * (1.0 - x);
}

double[][,] InitializeWeights(int[] layerSizes)
{
    Random rand = new Random();
    double[][,] weights = new double[layerSizes.Length - 1][,];

    for (int l = 0; l < layerSizes.Length - 1; l++)
    {
        weights[l] = new double[layerSizes[l], layerSizes[l + 1]];
        for (int i = 0; i < layerSizes[l]; i++)
        {
            for (int j = 0; j < layerSizes[l + 1]; j++)
            {
                weights[l][i, j] = rand.NextDouble() * 2.0 - 1.0;
            }
        }
    }
    return weights;
}


double[][] ForwardPass(double[] inputs, double[][,] weights)
{
    double[][] activations = new double[weights.Length + 1][];
    activations[0] = inputs;

    for (int l = 0; l < weights.Length; l++)
    {
        int outputSize = weights[l].GetLength(1);
        activations[l + 1] = new double[outputSize];

        for (int i = 0; i < outputSize; i++)
        {
            double sum = 0.0;
            for (int j = 0; j < activations[l].Length; j++)
            {
                sum += activations[l][j] * weights[l][j, i];
            }
            activations[l + 1][i] = Sigmoid(sum);
        }
    }

    return activations;
}

void Backpropagation(double[] inputs, double[] expectedOutput, double[][] activations, double[][,] weights, double learningRate)
{
    int outputLayer = weights.Length;
    double[] outputErrors = new double[weights[outputLayer - 1].GetLength(1)];
    double[] outputDeltas = new double[weights[outputLayer - 1].GetLength(1)];

    for (int i = 0; i < outputErrors.Length; i++)
    {
        outputErrors[i] = expectedOutput[i] - activations[outputLayer][i];
        outputDeltas[i] = outputErrors[i] * SigmoidDerivative(activations[outputLayer][i]);
    }

    double[][] errors = new double[weights.Length][];
    double[][] deltas = new double[weights.Length][];
    errors[outputLayer - 1] = outputErrors;
    deltas[outputLayer - 1] = outputDeltas;

    for (int l = weights.Length - 2; l >= 0; l--)
    {
        int layerSize = weights[l].GetLength(1);
        errors[l] = new double[layerSize];
        deltas[l] = new double[layerSize];

        for (int i = 0; i < layerSize; i++)
        {
            double error = 0.0;
            for (int j = 0; j < weights[l + 1].GetLength(1); j++)
            {
                error += deltas[l + 1][j] * weights[l + 1][i, j];
            }
            errors[l][i] = error;
            deltas[l][i] = error * SigmoidDerivative(activations[l + 1][i]);
        }
    }

    for (int l = weights.Length - 1; l >= 0; l--)
    {
        for (int i = 0; i < weights[l].GetLength(0); i++)
        {
            for (int j = 0; j < weights[l].GetLength(1); j++)
            {
                weights[l][i, j] += learningRate * deltas[l][j] * activations[l][i];
            }
        }
    }
}

void TrainNetwork(double[,] inputs, double[,] expectedOutputs, double[][,] weights, int epochs, double learningRate)
{
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        for (int i = 0; i < inputs.GetLength(0); i++)
        {
            double[] input = new double[inputs.GetLength(1)];
            for (int j = 0; j < inputs.GetLength(1); j++)
            {
                input[j] = inputs[i, j];
            }

            double[][] activations = ForwardPass(input, weights);

            double[] expectedOutput = new double[expectedOutputs.GetLength(1)];
            for (int j = 0; j < expectedOutputs.GetLength(1); j++)
            {
                expectedOutput[j] = expectedOutputs[i, j];
            }

            Backpropagation(input, expectedOutput, activations, weights, learningRate);
        }
    }
}

void Main()
{
    // Parameters for easy modification
    int[] layerSizes = { 2, 4, 4, 1 }; // Input layer, two hidden layers, and output layer
    int epochs = 10000;
    double learningRate = 0.1;

    // Example input and output data
    double[,] inputs = new double[,]
    {
        { 0, 0 },
        { 0, 1 },
        { 1, 0 },
        { 1, 1 }
    };

    double[,] expectedOutputs = new double[,]
    {
        { 0 },
        { 1 },
        { 1 },
        { 0 }
    };

    // Initialize weights
    double[][,] weights = InitializeWeights(layerSizes);

    // Train the network
    TrainNetwork(inputs, expectedOutputs, weights, epochs, learningRate);

    // Testing the network after training
    for (int i = 0; i < inputs.GetLength(0); i++)
    {
        double[] input = new double[inputs.GetLength(1)];
        for (int j = 0; j < inputs.GetLength(1); j++)
        {
            input[j] = inputs[i, j];
        }

        double[][] activations = ForwardPass(input, weights);
        Console.WriteLine($"Input: {input[0]}, {input[1]} => Output: {activations[activations.Length - 1][0]}");
    }
}