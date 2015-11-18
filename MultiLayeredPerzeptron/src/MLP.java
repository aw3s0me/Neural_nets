import java.io.*;
import java.util.ArrayList;
import java.util.Random;

/**
 * Created by aw3s0_000 on 04.11.2015.
 * Aleksandr Korovin and Andrei Zhukov. Uni Bonn. Neural nets assignment PB-A.
 */
public class MLP
{
    public class MLPLayer
    {
        private int nOutput;
        private int nInput;
        private double learnRate;
        private ITransferFunction transferFunction;
        private double[] input;
        private double[][] weights;
        private double[] error;
        private double[] output;
        private double[] net;

        public int getnOutput() {
            return nOutput;
        }

        public double[] getErrors() {
            return error;
        }

        public double[][] getWeights() {
            return weights;
        }

        public MLPLayer(int inputSize, int outputSize, double learnRate, ITransferFunction transferFunction, Random random) {
            this.learnRate = learnRate;
            this.transferFunction = transferFunction;
            this.nOutput = outputSize;
            this.nInput = inputSize;

            initializeWeights(random);
        }

        private double[] applyBias(double[] input, int nInput) {
            double[] newInput = new double[nInput + 1];

            for (int i = 0; i < input.length; i++) {
                newInput[i] = input[i];
            }
            newInput[nInput] = 1;
            //Initialization of bias (each element equals to 1)

            return newInput;
        }

        /**
         * get output of MLPLayer
         * @param input input vector
         */
        public double[] run(double[] input) {
            if (input == null)
            {
                throw new IllegalArgumentException("input matrix cant be null");
            }
            double[] newInput = applyBias(input, nInput);
            net = new double[nOutput];
            output = new double[nOutput];
            this.input = newInput;

            for (int i = 0; i < output.length; i++) {
                output[i] = 0.0;
                for (int j = 0; j < newInput.length; j++) {
                    output[i] += weights[j][i] * newInput[j];
                }
                net[i] = output[i];
                output[i] = transferFunction.calculate(output[i]);
            }

            return output;
        }

        private void initializeWeights(Random random) {
            weights = new double[nInput + 1][nOutput];

            for (int i = 0; i < weights.length; i++) {
                for (int j = 0; j < weights[i].length; j++) {
                    weights[i][j] = MLP.MIN_GEN_WEIGHT + (MLP.MAX_GEN_WEIGHT - MLP.MIN_GEN_WEIGHT) * random.nextDouble();
                }
            }
        }

        private double[] getOutputError(double[] teacher) {
            error = new double[nOutput];
            for (int i = 0; i < error.length; i++) {
                error[i] = (teacher[i] - output[i]) * transferFunction.calculateDerivative(net[i]);
            }

            return error;
        }

        private double[] getHiddenError(MLPLayer prevLayer) {
            error = new double[nOutput];
            double[] prevErrors = prevLayer.getErrors();
            double[][] prevWeights = prevLayer.getWeights();
            for (int i = 0; i < nOutput; i++) {
                double sum = 0;
                for (int j = 0; j < prevLayer.getnOutput(); j++) {
                    sum += prevWeights[i][j] * prevErrors[j];
                }
                error[i] = sum * transferFunction.calculateDerivative(net[i]);
            }

            return error;
        }

        public void setNewWeights() {
            for (int i = 0; i < nOutput; i++) {
                for (int j = 0; j < nInput; j++) {
                    weights[j][i] += learnRate * error[i] * input[j];
                }
            }

        }
    }

    //GENERATION CONSTANTS
    private static final double MIN_GEN_WEIGHT = -2;
    private static final double MAX_GEN_WEIGHT = 2;
    private static final int GEN_SEED = 10;
    private double learnRate;
    private final Random RANDOMIZER = new Random();
    //LAYERS
    private static final int HID_LAYER_NUM_1 = 6; //in case of 1-hidden-layer and 2-hidden layers.
    private static final int HID_LAYER_NUM_2 = 8; //in case of 2 hidden layers.
    private ArrayList<MLPLayer> layers = new ArrayList<>();
    //STI = Sigmoid (0 index), Tanh (1 index), Identity (2 index)
    private static final ArrayList<ITransferFunction> TRANSFER_FUNCTIONS_STI = new ArrayList<ITransferFunction>(){{
        add(new SigmoidFunction()); //index == 0
        add(new TanhFunction()); //index == 1
        add(new IdentityFunction()); //index == 2
    }};

    private void setRandomizer() {
        RANDOMIZER.setSeed(GEN_SEED);
    }

    /**
     * Function that creates 2-D array of fileInputs from file
     * File structure:
     * 1) nOutput - number of neurons - first line
     * 2) nPatterns - number of objects (,) nProperties - number of properties
     * @param nInput number of input neurons
     * @param nOutput number of output neurons
     * @param hiddenLayerSizes number of hidden neurons in each hidden layer
     * @param learnRate learning rate. if it is needed to set multiple learning rates. change to array
     */
    public MLP(int nInput, int nOutput, double learnRate, int[] hiddenLayerSizes)
    {
        this.learnRate = learnRate;
        ArrayList<Integer> layerSizes = setLayerSizes(nInput, nOutput, hiddenLayerSizes);
        initializeLayers(layerSizes);
    }

    private ArrayList<Integer> setLayerSizes(int nInput, int nOutput, int[] hiddenLayerSizes) {
        ArrayList<Integer> layerSizes = new ArrayList<>();
        layerSizes.add(nInput);

        if (hiddenLayerSizes != null && hiddenLayerSizes.length != 0) {
            for (int hiddenLayerSize : hiddenLayerSizes) {
                layerSizes.add(hiddenLayerSize);
            }
        }

        layerSizes.add(nOutput);
        return layerSizes;
    }

    private void initializeLayers(ArrayList<Integer> layerSizes) {
        if (layerSizes.size() > 4) {
            throw new IllegalArgumentException("Number of layers shouldnt be more than 4");
        }
        setRandomizer();
        for (int i = 1; i < layerSizes.size(); i++) {
            layers.add(new MLPLayer(layerSizes.get(i-1),
                    layerSizes.get(i), learnRate, TRANSFER_FUNCTIONS_STI.get(i - 1), RANDOMIZER));
        }
    }

    private void train(FileInput fileInput, int iterationsNum) throws FileNotFoundException, UnsupportedEncodingException {
        PrintWriter writer = new PrintWriter("learning.curve", "UTF-8");
        int nPatterns = fileInput.getnPatterns();
        double[][] inputs = fileInput.getFileInputs();
        double[][] teacher = fileInput.getTeacher();

        writer.println("# X     Y");
        for (int i = 0; i < iterationsNum; i++) {
            double objSum = 0;
            for (int j = 0; j < nPatterns; j++) {
                double[] output = inputs[j];
                double ErrSum = 0;
                for (MLPLayer layer : layers) {
                    output = layer.run(output);
                }
                backPropagation(teacher[j]);
                for (int k = 0; k < output.length; k++) {
                    double diff = (teacher[j][k] - output[k]);
                    ErrSum += diff * diff;
                }
                ErrSum = ErrSum * 0.5;
                objSum += ErrSum;
            }
            writer.println(String.format("%d %.3f", i, objSum));
        }

        writer.close();
    }

    /**
     * function to calculate back propagation
     */
    private void backPropagation(double[] desiredOutput) {
        double[] prevErrors = null;
        for (int i = layers.size() - 1; i >= 0; i--) {
            MLPLayer curLayer = layers.get(i);
            if (i == layers.size() - 1) {
                prevErrors = curLayer.getOutputError(desiredOutput);
            }
            else {
                if (prevErrors == null) {
                    throw new RuntimeException("Vector of errors from previous layer cannot be null");
                }

                prevErrors = curLayer.getHiddenError(layers.get(i+1));
            }
        }

        for (MLPLayer mlpLayer : layers) {
            mlpLayer.setNewWeights();
        }
    }

    public static void printMatrix(String introStr, double[][] matrix) {
        String str = "";
        int rows = matrix.length;

        System.out.println(introStr);

        if (matrix != null) {
            for (int i = 0; i < matrix.length; i++) {
                for (int j = 0; j < matrix[i].length; j++) {
                    System.out.print(matrix[i][j] + " ");
                }
                System.out.print("\n");
            }
        }
    }

    public static void main(String[] args) throws Exception {
        FileInput fileInput = new FileInput("training2.dat");

        MLP perzeptron = new MLP(fileInput.getnInput(), fileInput.getnOutput(), 0.1, new int[]{HID_LAYER_NUM_1, HID_LAYER_NUM_2});
        //Set hidden layers size = 1
        //MLP perzeptron = new MLP(fileInput.getnInput(), fileInput.getnOutput(), 0.1, new int[]{HID_LAYER_NUM_1});
        //Set hidden layers size = 0
        //MLP perzeptron = new MLP(fileInput.getnInput(), fileInput.getnOutput(), 0.1, new int[]{});
        perzeptron.train(fileInput, 500);
        printMatrix("Input matrix: ", fileInput.getFileInputs());
        printMatrix("Teacher matrix: ", fileInput.getTeacher());
        //uncomment when test
        FileInput testInput = new FileInput("test2.dat");
        perzeptron.test(testInput);
    }

    /**
     * To run test file with already trained neural net
     * @param fileInput file input containing matrices
     * @throws FileNotFoundException
     * @throws UnsupportedEncodingException
     */
    private void test(FileInput fileInput) throws FileNotFoundException, UnsupportedEncodingException {
        PrintWriter writer = new PrintWriter("test.result", "UTF-8");
        int nPatterns = fileInput.getnPatterns();
        double[][] inputs = fileInput.getFileInputs();
        double[][] teacher = fileInput.getTeacher();

        double objSum = 0;
        for (int j = 0; j < nPatterns; j++) {
            double[] output = inputs[j];
            double ErrSum = 0;
            for (MLPLayer layer : layers) {
                output = layer.run(output);
            }
            backPropagation(teacher[j]);
            //calculate error
            for (int k = 0; k < output.length; k++) {
                double diff = (teacher[j][k] - output[k]);
                ErrSum += diff * diff;
            }
            ErrSum = ErrSum * 0.5;
            objSum += ErrSum; //error for all patterns
            writer.println(String.format("Pattern %d error: %.6f", j, ErrSum));
        }
        writer.println(String.format("Final error (for all patterns): %.3f", objSum));

        writer.close();
    }

    /**
     * Created by aw3s0_000 on 03.11.2015.
     */
    public interface ITransferFunction
    {
        double calculate(double val);
        double calculateDerivative(double val);
    }

    public static class SigmoidFunction implements ITransferFunction
    {
        @Override
        public double calculate(double val)
        {
            return (1/( 1 + Math.pow(Math.E,(-1*val))));
        }

        @Override
        public double calculateDerivative(double val)
        {
            double funcVal = calculate(val);
            return funcVal * (1- funcVal);
        }
    }

    public static class TanhFunction implements ITransferFunction
    {
        @Override
        public double calculate(double val)
        {
            return Math.tanh(val);
        }

        @Override
        public double calculateDerivative(double val)
        {
            double funcVal = calculate(val);
            return 1 - (funcVal * funcVal);
        }
    }

    public static class IdentityFunction implements ITransferFunction
    {
        @Override
        public double calculate(double val) {
            return val;
        }

        @Override
        public double calculateDerivative(double val)
        {
            return 1;
        }
    }
}