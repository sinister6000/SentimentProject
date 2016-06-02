package cs517.data;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Scanner;
import java.util.regex.MatchResult;
import java.util.regex.Pattern;

/**
 * Created by allen on 5/18/2016.
 */

/**
 * Class to store a review. Unlabeled reviews get a polarity of -1. During the creation of a Review object,
 * we tokenize and translate to word2vec vectors.
 */





public class Review {


    private static int nextID = 0;

    String id;
    int score;
    int polarity;
    String reviewText;
    INDArray reviewVecs;


    public Review() {
    }

    /**
     * Constructs a Review from a tab delimited String.
     * @param parsedReview a review parsed from the Maas dataset
     */
    public Review(String parsedReview) {
        Scanner sc = new Scanner(parsedReview);
        Pattern p = Pattern.compile("\"(\\d+_(\\d+))\"\\t([01])?\\t?\"(.*)\"$");
        sc.findInLine(p);
        MatchResult result = sc.match();
        id = result.group(1);
        score = Integer.parseInt(result.group(2));
        reviewText = result.group(4);
        try {
            polarity = Integer.parseInt(result.group(3));
        } catch (NumberFormatException e) {
            polarity = -1;
        } finally {
            sc.close();
        }


    }
}
