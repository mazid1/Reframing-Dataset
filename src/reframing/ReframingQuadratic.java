package reframing;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

/**
 *
 * @author Mazid
 */
public class ReframingQuadratic {

    Instances train, test;
    Classifier model;

    // y = alpha x^2 + beta x + gamma
    public void selectAlphaBetaGamma(int idx) throws Exception {
        double p = 0.001;

        double alpha = 0.0;
        double negAlpha = alpha, posAlpha = alpha;
        Instances shiftedNegTest = new Instances(test);
        Instances shiftedPosTest = new Instances(test);

        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(model, test);
        double meanAbsoluteError = eval.meanAbsoluteError();
        double tmpMeanAbsoluteError = meanAbsoluteError;
        double negMeanAbsoluteError;
        double posMeanAbsoluteError;

        // decrease alpha for geting better result
        int count = 0;
        do {
            negAlpha -= p;
            negMeanAbsoluteError = tmpMeanAbsoluteError; // save better meanAbsoluteError
            for (int i = 0; i < shiftedNegTest.numInstances(); i++) {
                shiftedNegTest.instance(i).setValue(idx, negAlpha * test.instance(i).value(idx) * test.instance(i).value(idx));
            }
            // evaluate shifted test data
            eval.evaluateModel(model, shiftedNegTest);
            tmpMeanAbsoluteError = eval.meanAbsoluteError();
            count++;
        } while (tmpMeanAbsoluteError <= negMeanAbsoluteError && count < 10); // continue if new result is better than older

        // increase alpha for geting better result
        tmpMeanAbsoluteError = meanAbsoluteError;
        count = 0;
        do {
            posAlpha += p;
            posMeanAbsoluteError = tmpMeanAbsoluteError; // save better meanAbsoluteError
            for (int i = 0; i < shiftedPosTest.numInstances(); i++) {
                shiftedPosTest.instance(i).setValue(idx, posAlpha * test.instance(i).value(idx) * test.instance(i).value(idx));
            }
            // evaluate shifted test data
            eval.evaluateModel(model, shiftedPosTest);
            tmpMeanAbsoluteError = eval.meanAbsoluteError();
            count++;
        } while (tmpMeanAbsoluteError <= posMeanAbsoluteError && count < 10); // continue if new result is better than older

        // select best alpha
        if (negMeanAbsoluteError < posMeanAbsoluteError && negMeanAbsoluteError < meanAbsoluteError) {
            alpha = negAlpha + p;
            meanAbsoluteError = negMeanAbsoluteError;
            test = shiftedNegTest;
        } else if (posMeanAbsoluteError < negMeanAbsoluteError && posMeanAbsoluteError < meanAbsoluteError) {
            alpha = posAlpha - p;
            meanAbsoluteError = posMeanAbsoluteError;
            test = shiftedPosTest;
        }

        // now same procedure for beta
        negMeanAbsoluteError = meanAbsoluteError;
        posMeanAbsoluteError = meanAbsoluteError;
        tmpMeanAbsoluteError = meanAbsoluteError;

        shiftedNegTest = new Instances(test);
        shiftedPosTest = new Instances(test);

        double beta = 0.0;
        double negBeta = beta, posBeta = beta;
        p = 0.01;
        
        // decrease beta for geting better result
        count = 0;
        do {
            negBeta -= p;
            negMeanAbsoluteError = tmpMeanAbsoluteError; // save better meanAbsoluteError
            for (int i = 0; i < shiftedNegTest.numInstances(); i++) {
                shiftedNegTest.instance(i).setValue(idx, negBeta * test.instance(i).value(idx));
            }
            // evaluate shifted test data
            eval.evaluateModel(model, shiftedNegTest);
            tmpMeanAbsoluteError = eval.meanAbsoluteError();
            count++;
        } while (tmpMeanAbsoluteError <= negMeanAbsoluteError && count < 10); // continue if new result is better than older

        // increase beta for geting better result
        tmpMeanAbsoluteError = meanAbsoluteError;
        count = 0;
        do {
            posBeta += p;
            posMeanAbsoluteError = tmpMeanAbsoluteError; // save better meanAbsoluteError
            for (int i = 0; i < shiftedPosTest.numInstances(); i++) {
                shiftedPosTest.instance(i).setValue(idx, posBeta * test.instance(i).value(idx));
            }
            // evaluate shifted test data
            eval.evaluateModel(model, shiftedPosTest);
            tmpMeanAbsoluteError = eval.meanAbsoluteError();
            count++;
        } while (tmpMeanAbsoluteError <= posMeanAbsoluteError && count < 10); // continue if new result is better than older

        // select best beta
        if (negMeanAbsoluteError < posMeanAbsoluteError && negMeanAbsoluteError < meanAbsoluteError) {
            beta = negBeta + p;
            meanAbsoluteError = negMeanAbsoluteError;
            test = shiftedNegTest;
        } else if (posMeanAbsoluteError < negMeanAbsoluteError && posMeanAbsoluteError < meanAbsoluteError) {
            beta = posBeta - p;
            meanAbsoluteError = posMeanAbsoluteError;
            test = shiftedPosTest;
        }

        // now same procedure for gamma
        negMeanAbsoluteError = meanAbsoluteError;
        posMeanAbsoluteError = meanAbsoluteError;
        tmpMeanAbsoluteError = meanAbsoluteError;

        shiftedNegTest = new Instances(test);
        shiftedPosTest = new Instances(test);

        double gamma = 0.0;
        double negGamma = gamma, posGamma = gamma;
        p = 1.0;
        
        // decrease gamma for geting better result
        count = 0;
        do {
            negGamma -= p;
            negMeanAbsoluteError = tmpMeanAbsoluteError; // save better meanAbsoluteError
            for (int i = 0; i < shiftedNegTest.numInstances(); i++) {
                shiftedNegTest.instance(i).setValue(idx, negGamma + test.instance(i).value(idx));
            }
            // evaluate shifted test data
            eval.evaluateModel(model, shiftedNegTest);
            tmpMeanAbsoluteError = eval.meanAbsoluteError();
            count++;
        } while (tmpMeanAbsoluteError <= negMeanAbsoluteError && count < 10); // continue if new result is better than older

        // increase gamma for geting better result
        tmpMeanAbsoluteError = meanAbsoluteError;
        count = 0;
        do {
            posGamma += p;
            posMeanAbsoluteError = tmpMeanAbsoluteError; // save better meanAbsoluteError
            for (int i = 0; i < shiftedPosTest.numInstances(); i++) {
                shiftedPosTest.instance(i).setValue(idx, posGamma + test.instance(i).value(idx));
            }
            // evaluate shifted test data
            eval.evaluateModel(model, shiftedPosTest);
            tmpMeanAbsoluteError = eval.meanAbsoluteError();
            count++;
        } while (tmpMeanAbsoluteError <= posMeanAbsoluteError && count < 10); // continue if new result is better than older

        // select best gamma
        if (negMeanAbsoluteError < posMeanAbsoluteError && negMeanAbsoluteError < meanAbsoluteError) {
            gamma = negBeta + p;
            meanAbsoluteError = negMeanAbsoluteError;
            test = shiftedNegTest;
        } else if (posMeanAbsoluteError < negMeanAbsoluteError && posMeanAbsoluteError < meanAbsoluteError) {
            gamma = posGamma - p;
            meanAbsoluteError = posMeanAbsoluteError;
            test = shiftedPosTest;
        }

        // now shift dataset using learned alpha beta gamma
        for (int i = 0; i < this.test.numInstances(); i++) {
            this.test.instance(i).setValue(idx, this.test.instance(i).value(idx) * beta + gamma);
        }
    }

    public void hillClimbing() throws Exception {
        int numOfAttributes = test.numAttributes() - 1; // excluding class attribute
        // find optimum alpha beta gamma for each attribute
        for (int i = 0; i < numOfAttributes; i++) {
            selectAlphaBetaGamma(i);
        }
        // now evaluate the shifted test data set
        Evaluation eval = new Evaluation(this.train);
        eval.evaluateModel(this.model, this.test);
        System.out.println(eval.toSummaryString("\nResults for quadratic shifted dataset\n======\n", false));
    }

    public void reframing(Instances train, Instances test, Classifier model) throws Exception {
        this.train = new Instances(train);
        this.test = new Instances(test);
        this.model = model;

        // call hillclimbing
        hillClimbing();
    }
}
