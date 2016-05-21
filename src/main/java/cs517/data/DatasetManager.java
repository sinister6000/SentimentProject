package cs517.data;

import java.io.*;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

/**
 * Created by allen on 5/17/2016.
 */
public class DatasetManager {
    private Map<String, Review> reviews;

    public DatasetManager() {
        reviews = new HashMap<>();
    }

    /**
     * Imports reviews from a file.
     * @param f File of reviews
     */
    public void importData(File f) {
        Scanner sc = null;
        try {
            sc = new Scanner(f);
            String currentLine = sc.nextLine();   // skip over 1st header line
            Review currentReview;
            while (sc.hasNextLine()) {
                currentLine = sc.nextLine();
                currentReview = new Review(currentLine);
                reviews.put(currentReview.id, currentReview);
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } finally {
            if (sc != null) {
                sc.close();
            }
        }
    }


    /**
     * Outputs reviews to dir, 1 review per file.
     * Usage example: to output all positive reviews, first create the directory for the files.
     *
     *       DatasetManager allReviews = new DatasetManager();
     *       File dataFile = new File('PATH_TO_DATAFILE');
     *       allReviews.importData(dataFile);
     *       File directory = new File(dataFile.getParent() + "labeled/pos");
     *       directory.mkdirs();
     *       allReviews.toSeparateFiles(directory, 1);
     *
     * @param dir directory to write files
     * @param scoreType unlabeled (-1), negative (0), or positive (1)
     */
    public void toSeparateFiles(File dir, int scoreType) {
        if (!reviews.isEmpty()) {
            Review currentReview;
            for (String key: reviews.keySet()) {
                currentReview = reviews.get(key);
                if ((int) currentReview.score == scoreType) {
                    File f = new File(dir, currentReview.id);
                    try (BufferedWriter bw = new BufferedWriter(new FileWriter(f))) {
                        bw.write(currentReview.reviewText);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }

                }
            }
        }
    }
}
