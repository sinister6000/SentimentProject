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
            sc.useDelimiter(System.getProperty("line.separator"));
            String currentLine = sc.next();   // skip over 1st header line
            Review currentReview;
            while (sc.hasNext()) {
                currentLine = sc.next();
//                System.out.println(currentLine);
                currentReview = new Review(currentLine);
                reviews.put(currentReview.id, currentReview);
//                System.out.println(currentReview.id + "\t" + currentReview.score + "\t" + currentReview.reviewText);
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
    public void toSeparateFiles(String dir, int scoreType) {
        if (!reviews.isEmpty()) {
            for (String key : reviews.keySet()) {
                Review currentReview = reviews.get(key);
                if (currentReview.score == scoreType) {
                    File f = new File(dir + currentReview.id + ".txt");
                    try (BufferedWriter bw = new BufferedWriter(new FileWriter(f))) {
                        bw.write(currentReview.reviewText);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }
        }
    }

    public void toSingleFile(String fname) {
        if (!reviews.isEmpty()) {
            File f = new File(fname);
            try (BufferedWriter bw = new BufferedWriter(new FileWriter(f))) {
                for (String key : reviews.keySet()) {
                    Review currentReview = reviews.get(key);
                    bw.write(currentReview.reviewText);
                    bw.write("\n");
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }


    public static void main(String[] args) {
        DatasetManager dm = new DatasetManager();
        File df = new File("src/main/resources/movieData/maasDataset/labeledTrainData.tsv");
        dm.importData(df);
        File unlabeledDF = new File("src/main/resources/movieData/maasDataset/unlabeledTrainData.tsv");
        dm.importData(unlabeledDF);
//        File dir = new File(df.getParent() + "/paravec/");
//        System.out.println("dir path: " + dir.getPath());
//        dir.mkdirs();
//        dm.toSeparateFiles(dir.getPath(), 1);
//        dm.toSeparateFiles(dir.getPath(), 0);
//        dm.toSeparateFiles(dir.getPath(), -1);

        dm.toSingleFile("src/main/resources/movieData/maasDataset/allReviewText.txt");


    }
}
