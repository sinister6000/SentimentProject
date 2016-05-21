package cs517.data;

/**
 * Created by allen on 5/18/2016.
 */

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
        String[] tempReview = parsedReview.split("\t");
        id = tempReview[0];
        try {
            score = Integer.parseInt(tempReview[1]);
            reviewText = tempReview[2];
        } catch (NumberFormatException e) {
            score = -1;
            reviewText = tempReview[1];
        }
    }



}
