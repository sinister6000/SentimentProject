package cs517;

import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.StringCleaning;

/**
 * Created by allen on 5/25/2016.
 */
public class MySentencePreProcessor implements SentencePreProcessor {
    @Override
    public String preProcess(String sentence) {
        return StringCleaning.stripPunct(sentence).toLowerCase();
    }

}
