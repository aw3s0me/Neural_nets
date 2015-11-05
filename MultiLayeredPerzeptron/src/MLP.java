import java.io.*;
import java.util.ArrayList;
import java.util.LinkedList;
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
        private ITransferFunction transferFunction;
        private double[] input;
        private double[][] weights;
        private double[][] dweights;
        private double[] error;
        private double[] output;
        private double[] net;

        public int getnOutput() {
            return nOutput;
        }

        public int getInputNumber() {
            return nInput;
        }

        public double[] getErrors() {
            return error;
        }

        public double[][] getWeights() {
            return weights;
        }

        public MLPLayer(int inputSize, int outputSize, ITransferFunction transferFunction, Random random) {
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
            //Initialization of bias row (each element equals to 1

            return newInput;
        }

        /**
         * get output of MLPLayer
         * @param input
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
            dweights = new double[nInput + 1][nOutput];

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
                    weights[j][i] += LEARNING_RATE * error[i] * input[j];
                }
            }

        }
    }

    private static final double LEARNING_RATE = 0.01;
    private static final int ITERATIONS_NUM = 200;
    //GENERATION CONSTANTS
    private static final double MIN_GEN_WEIGHT = -2;
    private static final double MAX_GEN_WEIGHT = 2;
    private static final int GEN_SEED = 2;
    private int numOfPatternsGlobal;
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
    //MATRICES
    private double[][] fileInputs;
    private double[][] teacher; //matrix of teacher values
    private double[][] output; //result after sigmoid function

    public double[][] getFileInputs() {
        return fileInputs;
    }

    public double[][] getTeacher() {
        return teacher;
    }

    private void setRandomizer() {
        RANDOMIZER.setSeed(GEN_SEED);
    }

    /**
     * Function that creates 2-D array of fileInputs from file
     * File structure:
     * 1) nOutput - number of neurons - first line
     * 2) nPatterns - number of objects (,) nProperties - number of properties
     * @param filename
     * @param hiddenLayerSizes
     * @throws Exception
     */
    public MLP(String filename, int[] hiddenLayerSizes) throws IOException {
        InputStream stream = ClassLoader.getSystemResourceAsStream(filename);
        BufferedReader buffer = new BufferedReader(new InputStreamReader(stream));
        String line;
        LinkedList<ArrayList<Double>> tempInpList = new LinkedList<>();
        LinkedList<ArrayList<Double>> tempTeacherList = new LinkedList<>();

        while ((line = buffer.readLine()) != null) {
            if (line.contains("#")) {
                continue;
            }
            String[] vals = line.trim().split("    "); //to parse

            String[] inpVals = vals[0].split(" ");
            String[] teacherVals = vals[1].split(" ");
            ArrayList<Double> inputRow = new ArrayList<>();
            ArrayList<Double> teacherRow = new ArrayList<>();

            for (int i = 0; i < inpVals.length; i++) {
                if (!inpVals[i].isEmpty())
                    inputRow.add(Double.parseDouble(inpVals[i]));
            }

            for (int i = 0; i < teacherVals.length; i++) {
                if (!teacherVals[i].isEmpty())
                    teacherRow.add(Double.parseDouble(teacherVals[i]));
            }

            tempInpList.add(inputRow);
            tempTeacherList.add(teacherRow);
        }

        numOfPatternsGlobal = tempInpList.size();
        int nProperties = tempInpList.get(0).size();
        int nOutput = tempTeacherList.get(0).size();

        fileInputs = new double[numOfPatternsGlobal][nProperties];
        teacher = new double[numOfPatternsGlobal][nOutput];

        for (int i = 0; i < numOfPatternsGlobal; i++) {
            ArrayList<Double> list = tempInpList.get(i);
            for (int j = 0; j < nProperties; j++) {
                fileInputs[i][j] = list.get(j);
            }
        }

        for (int i = 0; i < numOfPatternsGlobal; i++) {
            ArrayList<Double> list = tempTeacherList.get(i);
            for (int j = 0; j < nOutput; j++) {
                teacher[i][j] = list.get(j);
            }
        }

        ArrayList<Integer> layerSizes = setLayerSizes(nProperties, nOutput, hiddenLayerSizes);

        initializeLayers(layerSizes);
    }

    private ArrayList<Integer> setLayerSizes(int nInput, int nOutput, int[] hiddenLayerSizes) {
        ArrayList<Integer> layerSizes = new ArrayList<>();
        layerSizes.add(nInput);

        if (hiddenLayerSizes != null || hiddenLayerSizes.length != 0) {
            for (int i = 0; i < hiddenLayerSizes.length; i++) {
                layerSizes.add(hiddenLayerSizes[i]);
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
                    layerSizes.get(i), TRANSFER_FUNCTIONS_STI.get(i - 1), RANDOMIZER));
        }
    }

    private void iteration() {
        for (int i = 0; i < ITERATIONS_NUM; i++) {
            for (int j = 0; j < numOfPatternsGlobal; j++) {
                double[] output = fileInputs[j];
                for (MLPLayer layer : layers) {
                    output = layer.run(output);
                }
                backPropagation(teacher[j]);
            }
        }
    }

    /**
     * function to calculate gradient descent
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

    public static void main(String[] args) throws Exception {
        MLP perzeptron = new MLP("training.dat", new int[]{HID_LAYER_NUM_1, HID_LAYER_NUM_2});
        Utils.printMatrix("Input matrix: ", perzeptron.getFileInputs());
        Utils.printMatrix("Teacher matrix: ", perzeptron.getTeacher());

        perzeptron.iteration();
    }

    /**
     * Created by aw3s0_000 on 03.11.2015.
     */
    public interface ITransferFunction {
        double calculate(double val);
        double calculateDerivative(double val);
    }

    public static class SigmoidFunction implements ITransferFunction {
        @Override
        public double calculate(double val) {
            return (1/( 1 + Math.pow(Math.E,(-1*val))));
        }

        @Override
        public double calculateDerivative(double val) {
            double funcVal = calculate(val);
            return funcVal * (1- funcVal);
        }
    }

    public static class TanhFunction implements ITransferFunction {
        @Override
        public double calculate(double val) {
            return Math.tanh(val);
        }

        @Override
        public double calculateDerivative(double val) {
            double funcVal = calculate(val);
            return 1 - (funcVal * funcVal);
        }
    }

    public static class IdentityFunction implements ITransferFunction {
        @Override
        public double calculate(double val) {
            return val;
        }

        @Override
        public double calculateDerivative(double val) {
            return 1;
        }
    }
}


