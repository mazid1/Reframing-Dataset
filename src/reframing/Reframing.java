package reframing;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

/**
 *
 * @author Mazid
 */
public class Reframing {
    
    Instances train, test;
    Classifier model;
    
    public void selectAlphaBeta(Instances train, Instances test, Classifier model, int idx) throws Exception {
        double p = 0.1;
        double alpha = 1.0;
        double negAlpha = alpha, posAlpha = alpha;
        Instances shiftedNegTest = test;
        Instances shiftedPosTest = test;
        
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(model, test);
        double meanAbsoluteError = eval.meanAbsoluteError();
        double tmpMeanAbsoluteError = meanAbsoluteError;
        double negMeanAbsoluteError;
        double posMeanAbsoluteError;
        
        // decrease alpha for geting better result
        do {
            negAlpha -= p;
            negMeanAbsoluteError = tmpMeanAbsoluteError; // save better meanAbsoluteError
            for(int i=0; i<shiftedNegTest.numInstances(); i++) {
                shiftedNegTest.instance(i).setValue(idx, negAlpha * test.instance(i).value(idx));		    	
            }
            // evaluate shifted test data
            eval.evaluateModel(model, shiftedNegTest);
            tmpMeanAbsoluteError = eval.meanAbsoluteError();
        }while(tmpMeanAbsoluteError < negMeanAbsoluteError); // continue if new result is better than older
        
        // increase alpha for geting better result
        tmpMeanAbsoluteError = meanAbsoluteError;
        do {
            posAlpha += p;
            posMeanAbsoluteError = tmpMeanAbsoluteError; // save better meanAbsoluteError
            for(int i=0; i<shiftedPosTest.numInstances(); i++) {
                shiftedPosTest.instance(i).setValue(idx, posAlpha * test.instance(i).value(idx));		    	
            }
            // evaluate shifted test data
            eval.evaluateModel(model, shiftedPosTest);
            tmpMeanAbsoluteError = eval.meanAbsoluteError();
        }while(tmpMeanAbsoluteError < posMeanAbsoluteError); // continue if new result is better than older
        
        // select best alpha
        
        if(negMeanAbsoluteError < posMeanAbsoluteError && negMeanAbsoluteError < meanAbsoluteError) {
            alpha = negAlpha + p;
        }
        else if(posMeanAbsoluteError < negMeanAbsoluteError && posMeanAbsoluteError < meanAbsoluteError) {
            alpha = posAlpha - p;
        }
        
    }
    
    public void hillClimbing(Instances train, Instances test, Classifier model) throws Exception {
        int numOfAttributes = test.numAttributes() - 1; // excluding class attribute
        // find optimum alpha beta for each attribute
        for(int i=0; i<numOfAttributes; i++) {
            selectAlphaBeta(train, test, model, i);
        }
    }
    
    public void reframing(Instances train, Instances test, Classifier model) throws Exception
    {
        this.train = train;
        this.test = test;
        this.model = model;
        
        // call hillclimbing	 
        hillClimbing(train, test, model);
    }
}
