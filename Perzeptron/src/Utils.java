import java.util.Iterator;
import java.util.List;

/**
 * Created by aw3s0_000 on 28.10.2015.
 */
public class Utils {
    /**
     * Print out 2darray
     * @param matrix
     */
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
}
