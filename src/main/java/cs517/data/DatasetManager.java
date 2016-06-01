package cs517.data;

import java.io.*;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

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


    public void toGroupFiles(String dir) {
        try {
            BufferedWriter b0 = new BufferedWriter(new FileWriter(new File(dir + "/0.tsv")));
            BufferedWriter b1 = new BufferedWriter(new FileWriter(new File(dir + "/1.tsv")));
            BufferedWriter b2 = new BufferedWriter(new FileWriter(new File(dir + "/2.tsv")));
            BufferedWriter b3 = new BufferedWriter(new FileWriter(new File(dir + "/3.tsv")));
            BufferedWriter b4 = new BufferedWriter(new FileWriter(new File(dir + "/4.tsv")));
            BufferedWriter b7 = new BufferedWriter(new FileWriter(new File(dir + "/7.tsv")));
            BufferedWriter b8 = new BufferedWriter(new FileWriter(new File(dir + "/8.tsv")));
            BufferedWriter b9 = new BufferedWriter(new FileWriter(new File(dir + "/9.tsv")));
            BufferedWriter b10 = new BufferedWriter(new FileWriter(new File(dir + "/10.tsv")));

            b0.write("id\treview\n");
            b1.write("id\tsentiment\treview\n");
            b2.write("id\tsentiment\treview\n");
            b3.write("id\tsentiment\treview\n");
            b4.write("id\tsentiment\treview\n");
            b7.write("id\tsentiment\treview\n");
            b8.write("id\tsentiment\treview\n");
            b9.write("id\tsentiment\treview\n");
            b10.write("id\tsentiment\treview\n");

            Review currRev;
            for (String revID : reviews.keySet()) {
                currRev = reviews.get(revID);
                Pattern p = Pattern.compile("\\d+_(\\d+)");
                Matcher matcher = p.matcher(currRev.id);
                int sentimentScore = Integer.parseInt(matcher.group(1));
                switch (sentimentScore) {
                    case 0:
                        b0.write(currRev.id + "\t" + currRev.reviewText + "\n");
                    case 1:
                        b1.write(currRev.id + "\t" + sentimentScore + "\t" + currRev.reviewText + "\n");
                    case 2:
                        b2.write(currRev.id + "\t" + sentimentScore + "\t" + currRev.reviewText + "\n");
                    case 3:
                        b3.write(currRev.id + "\t" + sentimentScore + "\t" + currRev.reviewText + "\n");
                    case 4:
                        b4.write(currRev.id + "\t" + sentimentScore + "\t" + currRev.reviewText + "\n");
                    case 7:
                        b7.write(currRev.id + "\t" + sentimentScore + "\t" + currRev.reviewText + "\n");
                    case 8:
                        b8.write(currRev.id + "\t" + sentimentScore + "\t" + currRev.reviewText + "\n");
                    case 9:
                        b9.write(currRev.id + "\t" + sentimentScore + "\t" + currRev.reviewText + "\n");
                    case 10:
                        b10.write(currRev.id + "\t" + sentimentScore + "\t" + currRev.reviewText + "\n");

                }
            }


        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    public static void main(String[] args) {
        DatasetManager dm = new DatasetManager();
        File df = new File("src/main/resources/movieData/maasDataset/labeledTrainData.tsv");
        dm.importData(df);
        File unlabeledDF = new File("src/main/resources/movieData/maasDataset/unlabeledTrainData.tsv");
        dm.importData(unlabeledDF);

        String pathForSplitData = "src/main/resources/movieData/maasDataset/splits/";
        File dir = new File(pathForSplitData);
        dir.mkdirs();
        dm.toGroupFiles(pathForSplitData);
//        File dir = new File(df.getParent() + "/paravec/");
//        System.out.println("dir path: " + dir.getPath());
//        dir.mkdirs();
//        dm.toSeparateFiles(dir.getPath(), 1);
//        dm.toSeparateFiles(dir.getPath(), 0);
//        dm.toSeparateFiles(dir.getPath(), -1);

        //dm.toSingleFile("src/main/resources/movieData/maasDataset/allReviewText.txt");


    }
}
