package cs517;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.PrefetchingSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Collection;

/**
 * Created by Renita on 5/24/16.
 */
public class Word2VecRawText {

    private static Logger log = LoggerFactory.getLogger(Word2VecRawText.class);

    public static void main(String[] args) throws Exception {

//        String filePath = new ClassPathResource("movieData/maasDataset/allReviewText.txt").getFile().getAbsolutePath();

        log.info("Load & Vectorize Sentences....");
        // Strip white space before and after for each line
        File f = new File("src/main/resources/movieData/maasDataset/allReviewText.txt");
        SentenceIterator iter = new PrefetchingSentenceIterator.Builder(new LineSentenceIterator(f))
              .setSentencePreProcessor(new MySentencePreProcessor())
              .build();
        // Split on white spaces in the line to get words
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        log.info("Building model....");
        Word2Vec vec = new Word2Vec.Builder()
              .minWordFrequency(5)
              .iterations(3)
              .epochs(4)
              .layerSize(100)
              .seed(42)
              .windowSize(5)
              .iterate(iter)
              .tokenizerFactory(t)
              .build();

        log.info("Fitting Word2Vec model....");
        vec.fit();

        log.info("Closest Words to \"awful\":");
        Collection<String> lst = vec.wordsNearest("awful", 10);
        System.out.println(lst);
        log.info("Closest Words to \"great\":");
        Collection<String> lst2 = vec.wordsNearest("great", 10);
        System.out.println(lst2);


        log.info("Writing word vectors to text file....");

        // Write word vectors
        WordVectorSerializer.writeWordVectors(vec, "src/main/resources/movieData/vectors/sentimentWordVectors.txt");


//        UiServer server = UiServer.getInstance();
//        System.out.println("Started on port " + server.getPort());
    }
}
