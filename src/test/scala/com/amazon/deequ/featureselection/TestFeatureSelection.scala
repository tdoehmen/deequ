/**
 * Copyright 2021 Logical Clocks AB. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not
 * use this file except in compliance with the License. A copy of the License
 * is located at
 *
 *     http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on
 * an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 * express or implied. See the License for the specific language governing
 * permissions and limitations under the License.
 *
 */

package com.amazon.deequ.featureselection

import com.amazon.deequ.SparkContextSpec
import com.amazon.deequ.analyzers.MutualInformation
import com.amazon.deequ.utils.FixtureSupport
import org.apache.spark.sql.DeequFunctions.stateful_kll_pmf
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions.{col, sum, when}
import org.scalatest.{Matchers, WordSpec}

import scala.math.log

class TestFeatureSelection extends WordSpec with Matchers with SparkContextSpec
  with FixtureSupport {

  "Approximate Mutual Information" should {

    "calculate Mutual Information based on KLL sketches" in
      withSparkSession { sparkSession =>
        val df = getDfWithNumericFractionalValuesForPMF(sparkSession)

        println(MutualInformation("att1", "target").calculate(df).value)
        println(MutualInformation("target", "att1").calculate(df).value)

        val resultX = df
          .agg(stateful_kll_pmf(col("att1"),
            bins = 30).alias("pmf"))
          .select("pmf")

        resultX.cache()
        resultX.show()
        val resultXRow = resultX.first()
        val rowX = resultXRow.getAs[Row]("pmf")
        val distX = rowX.getAs[Seq[Double]]("dist")
        val minX = rowX.getAs[Double]("min")
        val maxX = rowX.getAs[Double]("max")

        val resultXY = df
          .groupBy("target")
          .agg(stateful_kll_pmf(col("att1"),
            bins = 30,
            start = Some(minX),
            end = Some(maxX)).alias("att1_pmf"),
            sum(when(col("target").isNotNull, 1)
              .otherwise(0)).alias("count_y"))
          .withColumn("percent_y", col("count_y") /  sum("count_y").over())
          .select("target", "percent_y", "att1_pmf")

        resultXY.cache()
        resultXY.show()

        import sparkSession.implicits._

        val approxMI = resultXY.map( row => {
          val percentY = row.getAs[Double]("percent_y")
          val att1 = row.getAs[Row]("att1_pmf")
          val distXY = att1.getAs[Seq[Double]]("dist")
          val klDivergence = distXY.zipWithIndex.map {
            case (pX, i) => {
              if (distX(i) == 0 || pX.toDouble == 0) {
                0
              } else {
                pX * log(distX(i) / pX.toDouble)
              }
            }
          }.sum * (-1)
          percentY * klDivergence
        }: Double).reduce(_ + _)

        println(approxMI)


        /*
        def getDistribution()

        numRows = df.count("att1")
        distributionA =
        for kll in klls:

          distributionB = []
          for i in range(100):
            size = (kll.globalMax - kll.globalMin) / 100
            from = kll.globalMin + (size)*i
            to = kll.globalMin + (size)*(i+1)
            if i == 0:
              count = kll.qSketch.getRank(from) kll.qSketch.getRankExclusive(to)
            else:
              count = kll.qSketch.getRankExclusive(from) kll.qSketch.getRankExclusive(to)
             distributionB.append(count)/numRows

            val klDivergence =
         */

      }
  }
}
