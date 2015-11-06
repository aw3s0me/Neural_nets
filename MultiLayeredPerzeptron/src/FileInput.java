import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.LinkedList;

/**
 * Created by aw3s0_000 on 06.11.2015.
 */
public class FileInput
{
    private int nPatterns;
    private int nOutput;
    private int nInput;
    private double[][] fileInputs;
    private double[][] teacher; //matrix of teacher values

    public int getnPatterns() {
        return nPatterns;
    }

    public int getnOutput() {
        return nOutput;
    }

    public int getnInput() {
        return nInput;
    }

    public double[][] getFileInputs() {
        return fileInputs;
    }

    public double[][] getTeacher() {
        return teacher;
    }

    public FileInput(String filename) throws IOException {
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

            for (String inpVal : inpVals) {
                if (!inpVal.isEmpty())
                    inputRow.add(Double.parseDouble(inpVal));
            }

            for (String teacherVal : teacherVals) {
                if (!teacherVal.isEmpty())
                    teacherRow.add(Double.parseDouble(teacherVal));
            }

            tempInpList.add(inputRow);
            tempTeacherList.add(teacherRow);
        }

        nPatterns = tempInpList.size();
        nInput = tempInpList.get(0).size();
        nOutput = tempTeacherList.get(0).size();

        fileInputs = new double[nPatterns][nInput];
        teacher = new double[nPatterns][nOutput];

        for (int i = 0; i < nPatterns; i++) {
            ArrayList<Double> list = tempInpList.get(i);
            for (int j = 0; j < nInput; j++) {
                fileInputs[i][j] = list.get(j);
            }
        }

        for (int i = 0; i < nPatterns; i++) {
            ArrayList<Double> list = tempTeacherList.get(i);
            for (int j = 0; j < nOutput; j++) {
                teacher[i][j] = list.get(j);
            }
        }
    }
}
