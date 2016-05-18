package cs517.data;

/**
 * Created by allen on 5/18/2016.
 */
public class Review {

    private static int nextID = 0;

    private String id;
    private double score;
    private String reviewText;

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
            score = Double.parseDouble(tempReview[1]);
            reviewText = tempReview[2];
        } catch (NumberFormatException e) {
            reviewText = tempReview[1];
        }
    }



}
