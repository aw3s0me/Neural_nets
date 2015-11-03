import java.io.*;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Random;

/**
 * Created by aw3s0_000 on 04.11.2015.
 * Aleksandr Korovin and Andrei Zhukov. Uni Bonn. Neural nets assignment PB-A.
 */
public class MultiLayerPerzeptron {
    private int N; //number of properties from file
    private int M; //number of neurons from file
    private int K; //number of objects from file
    private static final double LEARNING_RATE = 0.01;
    private static final int ITERATIONS_NUM = 200;
    private static final double MIN_GEN_WEIGHT = -2;
    private static final double MAX_GEN_WEIGHT = 2;
    private static final int GEN_SEED = 2;
    //STI = Sigmoid (0 index), Tanh (1 index), Identity (2 index)
    private static final ArrayList<ITransferFunction> TRANSFER_FUNCTIONS_STI = new ArrayList<ITransferFunction>(){{
        add(new SigmoidFunction()); //index == 0
        add(new TanhFunction()); //index == 1
        add(new IdentityFunction()); //index == 2
    }};
    private final Random RANDOMIZER = new Random();
    private double[][] inputs;
    private double[][] teacher; //matrix of teacher values
    private double[][] hidden;
    private double[][] weights; //matrix of weights
    private double[][] weights2;
    private double[][] resultsMult; //result after multiplication of inputs on weights matrixes
    private double[][] sigmResult; //result after sigmoid function

    public int getNumberOfNeurons() {
        return M;
    }

    public int getNumberOfProperties() {
        return N;
    }

    public int getNumberOfObjects() {
        return K;
    }

    public double[][] getResultsMult() {
        return resultsMult;
    }

    public double[][] getInputs() {
        return inputs;
    }

    public double[][] getTeacher() {
        return teacher;
    }

    public double[][] getWeights() {
        return weights;
    }

    public void calcSigm() {
        sigmResult = new double[K][M];
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < M; j++) {
                sigmResult[i][j] = TRANSFER_FUNCTIONS_STI.get(0).calculate(resultsMult[i][j]);
            }
        }
    }

    private void setRandomizer() {
        RANDOMIZER.setSeed(GEN_SEED);
    }

    /**
     * Function that creates 2-D array of inputs from file
     * File structure:
     * 1) M - number of neurons - first line
     * 2) K - number of objects (,) N - number of properties
     * @param filename
     * @throws Exception
     */
    public MultiLayerPerzeptron(String filename) throws IOException {
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

        K = tempInpList.size();
        N = tempInpList.get(0).size();
        M = tempTeacherList.get(0).size();

        inputs = new double[K][N + 1];
        teacher = new double[K][M];

        for (int i = 0; i < K; i++) {
            ArrayList<Double> list = tempInpList.get(i);
            for (int j = 0; j < N; j++) {
                inputs[i][j] = list.get(j);
            }
        }

        //Initialization of bias row (each element equals to 1
        for (int i = 0; i < K; i++) {
            inputs[i][N] = 1;
        }

        for (int i = 0; i < K; i++) {
            ArrayList<Double> list = tempTeacherList.get(i);
            for (int j = 0; j < M; j++) {
                teacher[i][j] = list.get(j);
            }
        }

        setRandomizer();
    }

    /**
     * function to calculate gradient descent
     */
    private void backPropagation() {
        double[][] transposedInputs = Matrix.transpose(inputs); //need to transpose input matrix, because it doesn't match dimensions
        double[][] errorResult = new double[K][M]; //matrix to store intermediate result

        for (int i = 0; i < K; i++) {
            for (int j = 0; j < M; j++) {
                errorResult[i][j] = LEARNING_RATE *
                        (teacher[i][j] - sigmResult[i][j]) *
                        (1 - sigmResult[i][j]) *
                        sigmResult[i][j];
            }
        }
        //get new weights
        weights = Matrix.add(weights, Matrix.multiply(transposedInputs, errorResult));
    }

    /**
     * To generate weight matrix
     */
    private void generateWeights() {
        weights = new double[N+1][M];
        Random r = RANDOMIZER;

        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                weights[i][j] = MIN_GEN_WEIGHT + (MAX_GEN_WEIGHT - MIN_GEN_WEIGHT) * r.nextDouble();
            }
        }
    }

    public void multiplyWeightsAndInputs() {
        resultsMult = Matrix.multiply(inputs, weights);
    }

    public double[][] getError() {
        return Matrix.subtract(teacher, sigmResult);
    }

    public static void main(String[] args) throws Exception {
        MultiLayerPerzeptron perzeptron = new MultiLayerPerzeptron("training.dat");

        System.out.println("Number of objects: " + perzeptron.getNumberOfObjects() + "\n");
        System.out.println("Number of neurons: " + perzeptron.getNumberOfNeurons() + "\n");
        System.out.println("Number of properties: " + perzeptron.getNumberOfProperties() + "\n");

        Utils.printMatrix("Input matrix: ", perzeptron.getInputs());
        Utils.printMatrix("Teacher matrix: ", perzeptron.getTeacher());

        perzeptron.generateWeights();

        for (int i = 0; i < ITERATIONS_NUM; i++) {
            perzeptron.multiplyWeightsAndInputs();
            perzeptron.calcSigm();
            perzeptron.backPropagation();
        }

        //Utils.printMatrix("Sigmoid result: ", perzeptron.getSigmResult());
        Utils.printMatrix("New weights: ", perzeptron.getWeights());
        Utils.printMatrix("Error matrix: ", perzeptron.getError());
    }

    /**
     * Created by aw3s0_000 on 03.11.2015.
     */
    public interface ITransferFunction {
        double calculate(double val);
    }

    public static class SigmoidFunction implements ITransferFunction {
        @Override
        public double calculate(double val) {
            return (1/( 1 + Math.pow(Math.E,(-1*val))));
        }
    }

    public static class TanhFunction implements ITransferFunction {
        @Override
        public double calculate(double val) {
            return Math.tanh(val);
        }
    }

    public static class IdentityFunction implements ITransferFunction {
        @Override
        public double calculate(double val) {
            return val;
        }
    }
}


