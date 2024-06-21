Console.WriteLine("Running Claude 3.5 Sonnet");
NeuralNetwork.Main();
// Done in 5 rounds including deleted context
// It keeps going off the rails, but once it gets it right, it gets it right

class NeuralNetwork
{
    const int INPUT_NODES = 2;
    static readonly int[] HIDDEN_LAYERS = { 4, 3 }; // Two hidden layers with 4 and 3 nodes respectively
    const int OUTPUT_NODES = 1;
    const double LEARNING_RATE = 0.1;
    const double MOMENTUM = 0.9;
    const int EPOCHS = 10000;

    static double[] inputLayer = new double[INPUT_NODES];
    static double[][] hiddenLayers;
    static double[] outputLayer = new double[OUTPUT_NODES];

    static double[][][] weights;
    static double[][] biases;

    static double[][][] prevDeltaWeights;

    static void InitializeWeights()
    {
        Random random = new Random();

        weights = new double[HIDDEN_LAYERS.Length + 1][][];
        biases = new double[HIDDEN_LAYERS.Length + 1][];
        prevDeltaWeights = new double[HIDDEN_LAYERS.Length + 1][][];
        hiddenLayers = new double[HIDDEN_LAYERS.Length][];

        // Input to first hidden layer
        weights[0] = new double[INPUT_NODES][];
        prevDeltaWeights[0] = new double[INPUT_NODES][];
        for (int i = 0; i < INPUT_NODES; i++)
        {
            weights[0][i] = new double[HIDDEN_LAYERS[0]];
            prevDeltaWeights[0][i] = new double[HIDDEN_LAYERS[0]];
            for (int j = 0; j < HIDDEN_LAYERS[0]; j++)
            {
                weights[0][i][j] = random.NextDouble() - 0.5;
            }
        }

        // Hidden layers
        for (int l = 0; l < HIDDEN_LAYERS.Length; l++)
        {
            hiddenLayers[l] = new double[HIDDEN_LAYERS[l]];
            biases[l] = new double[HIDDEN_LAYERS[l]];
            for (int i = 0; i < HIDDEN_LAYERS[l]; i++)
            {
                biases[l][i] = random.NextDouble() - 0.5;
            }

            if (l < HIDDEN_LAYERS.Length - 1)
            {
                weights[l + 1] = new double[HIDDEN_LAYERS[l]][];
                prevDeltaWeights[l + 1] = new double[HIDDEN_LAYERS[l]][];
                for (int i = 0; i < HIDDEN_LAYERS[l]; i++)
                {
                    weights[l + 1][i] = new double[HIDDEN_LAYERS[l + 1]];
                    prevDeltaWeights[l + 1][i] = new double[HIDDEN_LAYERS[l + 1]];
                    for (int j = 0; j < HIDDEN_LAYERS[l + 1]; j++)
                    {
                        weights[l + 1][i][j] = random.NextDouble() - 0.5;
                    }
                }
            }
        }

        // Last hidden layer to output
        int lastHiddenLayer = HIDDEN_LAYERS.Length - 1;
        weights[lastHiddenLayer + 1] = new double[HIDDEN_LAYERS[lastHiddenLayer]][];
        prevDeltaWeights[lastHiddenLayer + 1] = new double[HIDDEN_LAYERS[lastHiddenLayer]][];
        for (int i = 0; i < HIDDEN_LAYERS[lastHiddenLayer]; i++)
        {
            weights[lastHiddenLayer + 1][i] = new double[OUTPUT_NODES];
            prevDeltaWeights[lastHiddenLayer + 1][i] = new double[OUTPUT_NODES];
            for (int j = 0; j < OUTPUT_NODES; j++)
            {
                weights[lastHiddenLayer + 1][i][j] = random.NextDouble() - 0.5;
            }
        }

        biases[HIDDEN_LAYERS.Length] = new double[OUTPUT_NODES];
        for (int i = 0; i < OUTPUT_NODES; i++)
        {
            biases[HIDDEN_LAYERS.Length][i] = random.NextDouble() - 0.5;
        }
    }

    static double Sigmoid(double x)
    {
        return 1 / (1 + Math.Exp(-x));
    }

    static double SigmoidDerivative(double x)
    {
        return x * (1 - x);
    }

    static void ForwardPropagate(double[] inputs)
    {
        // Input to first hidden layer
        for (int i = 0; i < HIDDEN_LAYERS[0]; i++)
        {
            double sum = biases[0][i];
            for (int j = 0; j < INPUT_NODES; j++)
            {
                sum += inputs[j] * weights[0][j][i];
            }
            hiddenLayers[0][i] = Sigmoid(sum);
        }

        // Between hidden layers
        for (int l = 1; l < HIDDEN_LAYERS.Length; l++)
        {
            for (int i = 0; i < HIDDEN_LAYERS[l]; i++)
            {
                double sum = biases[l][i];
                for (int j = 0; j < HIDDEN_LAYERS[l - 1]; j++)
                {
                    sum += hiddenLayers[l - 1][j] * weights[l][j][i];
                }
                hiddenLayers[l][i] = Sigmoid(sum);
            }
        }

        // Last hidden layer to output
        for (int i = 0; i < OUTPUT_NODES; i++)
        {
            double sum = biases[HIDDEN_LAYERS.Length][i];
            for (int j = 0; j < HIDDEN_LAYERS[HIDDEN_LAYERS.Length - 1]; j++)
            {
                sum += hiddenLayers[HIDDEN_LAYERS.Length - 1][j] * weights[HIDDEN_LAYERS.Length][j][i];
            }
            outputLayer[i] = Sigmoid(sum);
        }
    }

    static void Backpropagate(double[] targets)
    {
        // Calculate output layer errors
        double[] outputErrors = new double[OUTPUT_NODES];
        for (int i = 0; i < OUTPUT_NODES; i++)
        {
            outputErrors[i] = (targets[i] - outputLayer[i]) * SigmoidDerivative(outputLayer[i]);
        }

        // Calculate hidden layer errors
        double[][] hiddenErrors = new double[HIDDEN_LAYERS.Length][];
        for (int l = HIDDEN_LAYERS.Length - 1; l >= 0; l--)
        {
            hiddenErrors[l] = new double[HIDDEN_LAYERS[l]];
            for (int i = 0; i < HIDDEN_LAYERS[l]; i++)
            {
                double error = 0;
                if (l == HIDDEN_LAYERS.Length - 1)
                {
                    for (int j = 0; j < OUTPUT_NODES; j++)
                    {
                        error += outputErrors[j] * weights[l + 1][i][j];
                    }
                }
                else
                {
                    for (int j = 0; j < HIDDEN_LAYERS[l + 1]; j++)
                    {
                        error += hiddenErrors[l + 1][j] * weights[l + 1][i][j];
                    }
                }
                hiddenErrors[l][i] = error * SigmoidDerivative(hiddenLayers[l][i]);
            }
        }

        // Update weights and biases
        for (int l = HIDDEN_LAYERS.Length; l >= 0; l--)
        {
            int fromNodes = l == 0 ? INPUT_NODES : HIDDEN_LAYERS[l - 1];
            int toNodes = l == HIDDEN_LAYERS.Length ? OUTPUT_NODES : HIDDEN_LAYERS[l];

            for (int i = 0; i < fromNodes; i++)
            {
                for (int j = 0; j < toNodes; j++)
                {
                    double input = l == 0 ? inputLayer[i] : hiddenLayers[l - 1][i];
                    double error = l == HIDDEN_LAYERS.Length ? outputErrors[j] : hiddenErrors[l][j];
                    double delta = LEARNING_RATE * error * input + MOMENTUM * prevDeltaWeights[l][i][j];
                    weights[l][i][j] += delta;
                    prevDeltaWeights[l][i][j] = delta;
                }
            }

            for (int i = 0; i < toNodes; i++)
            {
                double error = l == HIDDEN_LAYERS.Length ? outputErrors[i] : hiddenErrors[l][i];
                biases[l][i] += LEARNING_RATE * error;
            }
        }
    }

    static void Train(double[][] trainingInputs, double[][] trainingOutputs, int epochs)
    {
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            double totalError = 0;
            for (int i = 0; i < trainingInputs.Length; i++)
            {
                inputLayer = trainingInputs[i];
                ForwardPropagate(inputLayer);
                Backpropagate(trainingOutputs[i]);

                // Calculate error
                for (int j = 0; j < OUTPUT_NODES; j++)
                {
                    totalError += Math.Pow(trainingOutputs[i][j] - outputLayer[j], 2);
                }
            }

            // Print error every 1000 epochs
            if (epoch % 1000 == 0)
            {
                Console.WriteLine($"Epoch {epoch}: Error = {totalError}");
            }
        }
    }

    public static void Main()
    {
        InitializeWeights();

        double[][] trainingInputs = new double[][]
        {
            new double[] {0, 0},
            new double[] {0, 1},
            new double[] {1, 0},
            new double[] {1, 1}
        };

        double[][] trainingOutputs = new double[][]
        {
            new double[] {0},
            new double[] {1},
            new double[] {1},
            new double[] {0}
        };

        Train(trainingInputs, trainingOutputs, EPOCHS);

        // Test the trained network
        for (int i = 0; i < trainingInputs.Length; i++)
        {
            ForwardPropagate(trainingInputs[i]);
            Console.WriteLine($"Input: {trainingInputs[i][0]}, {trainingInputs[i][1]} | Output: {outputLayer[0]}");
        }
    }
}