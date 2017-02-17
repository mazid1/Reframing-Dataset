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
    int num;

    // y = alpha x^2 + beta x + gamma
    public void selectAlphaBetaGamma(int idx) throws Exception {
     
        double alpha=0, beta=1, gamma=0;
        Instances tmpTest = new Instances(test);

        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(model, test);
        double maxCorrect = eval.correct();
        double tmpCorrect = maxCorrect;
        //double minMeanAbsoluteError = meanAbsoluteError;
        
        for(double a = -0.009; a<=0.01; a+=0.001) {
            for(double b = -0.09; b<=0.09; b+=0.01) {
                for(double g = -9; g<=9; g+=1) {
                    for (int i = 0; i < num; i++) {
                        tmpTest.instance(i).setValue(idx, a * test.instance(i).value(idx) * test.instance(i).value(idx)
                                                            + b * test.instance(i).value(idx)
                                                            + g);
                    }
                    eval = new Evaluation(train);
                    eval.evaluateModel(model, tmpTest);
                    tmpCorrect = eval.correct();
                    if(tmpCorrect > maxCorrect) {
                        maxCorrect = tmpCorrect;
                        alpha = a;
                        beta = b;
                        gamma = g;
                        //System.out.println("Correct now = " + eval.correct());
                    }
                    
                }
            }
        }
        
        // now shift dataset using learned alpha beta gamma
        //System.out.println("Alpha=" + alpha + " beta=" + beta + " gamma=" + gamma);
        //System.out.println(this.test.instance(0));
        //if(alpha != 0 || beta != 0 || gamma != 0) {
            for (int i = 0; i < num; i++) {
                //if(i==0) System.out.println("before shift" + this.test.instance(0));
                this.test.instance(i).setValue(idx, this.test.instance(i).value(idx) * this.test.instance(i).value(idx) * alpha 
                                                    + this.test.instance(i).value(idx) * beta
                                                    + gamma);
            }
        //}
        
        //System.out.println("Alpha=" + alpha + " beta=" + beta + " gamma=" + gamma);
        //System.out.println("after shift" + this.test.instance(0));
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
        System.out.println(eval.toSummaryString("\nResults for quadratic shifted dataset\n" + "Shifted " + num + " data\n======================\n", false));
        //System.out.println("Precision: " + eval.precision(this.test.classIndex()) + "\nRecall: " + eval.recall(this.test.classIndex()));
        //System.out.println("Correct = " + eval.correct());
    }

    public void reframing(Instances train, Instances test, Classifier model, int num) throws Exception {
        this.train = new Instances(train);
        this.test = new Instances(test);
        this.model = model;
        this.num = num;
        //System.out.println(this.test.instance(0));
        // call hillclimbing
        hillClimbing();
    }
}
