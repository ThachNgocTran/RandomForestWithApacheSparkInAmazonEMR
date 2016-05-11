/**
  * Created on 09/May/2016.
  * Reference: https://spark.apache.org/docs/latest/mllib-ensembles.html#gradient-boosted-trees-vs-random-forests
  */

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.MLUtils

object MyApp {
    def main(args : Array[String]) {
        try {
            println("*** PROGRAM STARTED ***")

            // for local testing
            //val cfg = new SparkConf().setMaster("local[2]").setAppName("RandomForestWithSparkInAmazonEMR")

            // for Amazon EMR
            val cfg = new SparkConf().setAppName("RandomForestWithSparkInAmazonEMR")

            val sc = new SparkContext(cfg)

            // for local testing
            //val data = MLUtils.loadLibSVMFile(sc, "C:\\Users\\iRobot\\Desktop\\libdata.data.txt.gz")

            // for Amazon EMR
            val data = MLUtils.loadLibSVMFile(sc, "s3://thachngoctranmyjar/libdata.data.txt.gz")

            val splits = data.randomSplit(Array(0.7, 0.3))
            val (trainingData, testData) = (splits(0), splits(1))

            // Train a RandomForest model.
            val numClasses = 7                              // We have 7 forest cover types.
            var categoricalFeaturesInfo = Map[Int, Int]()   // Indicate which columns are of categorical data.

            // we have 44 categorical features in the dataset ("True/False" data type)
            var x = 0
            for (x <- 10 to 53){
                categoricalFeaturesInfo += (x -> 2)
            }

            val numTrees = 3                                // In reality, they can even build 1000 trees!
            val featureSubsetStrategy = "auto"              // Let the algorithm choose.
            val impurity = "gini"                           // The other is "entropy".
            val maxDepth = 10                               // The dept of the classification trees. Too short ==> increase bias; too deep ==> increase variance (overfitting!)
            val maxBins = 32

            // See: https://github.com/apache/spark/blob/master/mllib/src/main/scala/org/apache/spark/mllib/tree/RandomForest.scala
            val model = RandomForest.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
                numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

            // Evaluate model on test instances and compute test error
            val labelAndPreds = testData.map { point =>
                val prediction = model.predict(point.features)
                (point.label, prediction)
            }
            val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count()

            println("Test Error = " + testErr)
            println("Learned classification forest model:\n" + model.toDebugString)

            // Save and load model
            //model.save(sc, "target/tmp/myRandomForestClassificationModel")
            //val sameModel = RandomForestModel.load(sc, "target/tmp/myRandomForestClassificationModel")

            println("*** PROGRAM SUCCEEDED ***")
        }
        catch   // this is the outer-most layer; so reasonably, we have "catch-all" exception.
            {
                case e: Exception => println("*** PROGRAM EXCEPTION: " + e.getMessage() + " ***");
            }
        finally
        {
            println("*** PROGRAM ENDED ***")
        }
    }
}
