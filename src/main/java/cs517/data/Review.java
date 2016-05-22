package cs517.data;

/**
 * Created by allen on 5/18/2016.
 */

import java.util.Scanner;
import java.util.regex.MatchResult;
import java.util.regex.Pattern;

/**
 * Class to store a review. Unlabeled reviews get a sccore of -1.
 */
public class Review {

    private static int nextID = 0;

    String id;
    int score;
    String reviewText;

    public Review() {
        id = Integer.toString(nextID++);
    }

    /**
     * Constructs a Review from a tab delimited String.
     * @param parsedReview a review parsed from the Maas dataset
     */
    public Review(String parsedReview) {
        Scanner sc = new Scanner(parsedReview);

        Pattern p = Pattern.compile("\"(\\d+_\\d+)\" +([01])? +\"(.*)\"");
        sc.findInLine(p);
        MatchResult result = sc.match();
        id = result.group(1);
        try {
            score = Integer.parseInt(result.group(2));
            reviewText = result.group(3);
        } catch (NumberFormatException e) {
            score = -1;
            reviewText = result.group(2);
        } finally {
            sc.close();
        }
    }
}
