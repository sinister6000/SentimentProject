Stanford MAAS dataset.
I think we want to store our data in a similar format, or even in identical format.
We can just parse straight from these files, using their IDs.

class Review {
    String id;
    int score;
    String reviewText;
}
