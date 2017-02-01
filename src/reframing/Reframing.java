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
            meanAbsoluteError = negMeanAbsoluteError;
            test = shiftedNegTest;
        }
        else if(posMeanAbsoluteError < negMeanAbsoluteError && posMeanAbsoluteError < meanAbsoluteError) {
            alpha = posAlpha - p;
            meanAbsoluteError = posMeanAbsoluteError;
            test = shiftedPosTest;
        }
        
        // now same procedure for beta
        negMeanAbsoluteError = meanAbsoluteError;
        posMeanAbsoluteError = meanAbsoluteError;
        tmpMeanAbsoluteError = meanAbsoluteError;
        
        shiftedNegTest = test;
        shiftedPosTest = test;
        
        double beta = 0.0;
        double negBeta = beta, posBeta = beta;
        
        // decrease beta for geting better result
        do {
            negBeta -= p;
            negMeanAbsoluteError = tmpMeanAbsoluteError; // save better meanAbsoluteError
            for(int i=0; i<shiftedNegTest.numInstances(); i++) {
                shiftedNegTest.instance(i).setValue(idx, negBeta + test.instance(i).value(idx));		    	
            }
            // evaluate shifted test data
            eval.evaluateModel(model, shiftedNegTest);
            tmpMeanAbsoluteError = eval.meanAbsoluteError();
        }while(tmpMeanAbsoluteError < negMeanAbsoluteError); // continue if new result is better than older
        
        // increase beta for geting better result
        tmpMeanAbsoluteError = meanAbsoluteError;
        do {
            posBeta += p;
            posMeanAbsoluteError = tmpMeanAbsoluteError; // save better meanAbsoluteError
            for(int i=0; i<shiftedPosTest.numInstances(); i++) {
                shiftedPosTest.instance(i).setValue(idx, posBeta + test.instance(i).value(idx));		    	
            }
            // evaluate shifted test data
            eval.evaluateModel(model, shiftedPosTest);
            tmpMeanAbsoluteError = eval.meanAbsoluteError();
        }while(tmpMeanAbsoluteError < posMeanAbsoluteError); // continue if new result is better than older
        
        // select best beta
        if(negMeanAbsoluteError < posMeanAbsoluteError && negMeanAbsoluteError < meanAbsoluteError) {
            beta = negBeta + p;
            meanAbsoluteError = negMeanAbsoluteError;
            test = shiftedNegTest;
        }
        else if(posMeanAbsoluteError < negMeanAbsoluteError && posMeanAbsoluteError < meanAbsoluteError) {
            beta = posBeta - p;
            meanAbsoluteError = posMeanAbsoluteError;
            test = shiftedPosTest;
        }
        
        // now shift dataset using learned alpha beta
        for(int i=0; i<this.test.numInstances(); i++) {
            this.test.instance(i).setValue(idx, this.test.instance(i).value(idx)*alpha+beta );		    	
        }
        //System.out.println("alpha = " + alpha + " beta = " + beta);
    }
    
    public void hillClimbing(Instances train, Instances test, Classifier model) throws Exception {
        int numOfAttributes = test.numAttributes() - 1; // excluding class attribute
        // find optimum alpha beta for each attribute
        for(int i=0; i<numOfAttributes; i++) {
            selectAlphaBeta(this.train, this.test, this.model, i);
        }
        // now evaluate the shifted test data set
        Evaluation eval = new Evaluation(this.train);
        eval.evaluateModel(this.model, this.test);
        System.out.println(eval.toSummaryString("\nResults for shifted dataset\n======\n", false));
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
