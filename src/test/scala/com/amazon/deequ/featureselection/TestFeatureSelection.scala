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
import com.amazon.deequ.analyzers.runners.{ByteQuantileNonSample, DoubleQuantileNonSample, FloatQuantileNonSample, IntQuantileNonSample, LongQuantileNonSample, ShortQuantileNonSample, UntypedQuantileNonSample}
import com.amazon.deequ.analyzers.{Histogram, KLLParameters, KLLSketch, KLLState, MutualInformation}
import com.amazon.deequ.utils.FixtureSupport
import org.apache.datasketches.frequencies.{ErrorType, ItemsSketch, LongsSketch}
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.feature.MrmrSelector
import org.apache.spark.sql.DeequFunctions.{stateful_kll_2, stateful_kll_agg}
import org.apache.spark.sql.{Row, functions}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{ByteType, DoubleType, FloatType, IntegerType, LongType, ShortType, StringType, StructType}
import org.scalatest.{Matchers, WordSpec}
import org.apache.spark.ml.feature.{StringIndexer, StringIndexerModel, VectorAssembler}
import org.apache.spark.mllib.linalg.{SparseVector, DenseVector => DenseVectorMLLib}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.expressions.XxHash64Function
import org.apache.spark.sql.catalyst.expressions.aggregate.{BaseFreqSketchImpl, FreqSketch, FrequentItemSketchHelpers, LongFreqSketchImpl, StringFreqSketchImpl}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.unsafe.types.UTF8String

import scala.annotation.switch

class TestFeatureSelection extends WordSpec with Matchers with SparkContextSpec
  with FixtureSupport {

  "Approximate Mutual Information" should {

    "calculate MRMR with fastmrmr" in
      withSparkSession { sparkSession =>
        // create test dataset
        val nRows = 10000
        val nVal = 1000
        val nTargetBins = 100
        val nUniques = 255
        val globalMaxNumeric = 1000.0

        var df = sparkSession.read.format("parquet")
          .load(f"test-data/features_int_$nVal.parquet")
        df = df.withColumn("target",
          functions.round(functions.rand(10) * lit(nTargetBins)))
        println(df.rdd.getNumPartitions)
        df.cache()
        df.count()

        //val x = XxHash64Function.hash(v, child.dataType, 42L)
        //LongFreqSketchImpl

        val t = System.nanoTime

        val selectedColumns = (1 to nVal).map(i => f"att$i").toArray :+ "target"
        val outputIndexes = selectedColumns.zipWithIndex
        val selectedIndexes = df.schema.fields
          .map { _.name }
          .zipWithIndex
          .toMap
        val selectedTypes = df.schema
          .filter( info => selectedColumns.contains(info.name) )
          .map { info => info.name -> info.dataType }
          .toMap

        // create hashes
        val hashedTable: RDD[Row] = df.rdd.map (
          row => {
            val values = row
            val outputs = selectedColumns.map { column =>
              val dfIdx = selectedIndexes(column)
              val dataType = selectedTypes(column)
              val valueInColumn = (dataType: @switch) match {
                case StringType  => {
                  if (values.isNullAt(dfIdx)) {
                    0
                  } else {
                    XxHash64Function.hash(UTF8String.fromString(values.getString(dfIdx)),
                      dataType, 42L)
                  }
                } case _  => {
                  values.get(dfIdx)
                }
              }
              valueInColumn
            }.toSeq
            Row(outputs: _*)
          })

        val hashingColumns = outputIndexes.filter(kv => selectedTypes(kv._1)
          == StringType).toMap

        println(hashingColumns)

        val sketching = FrequentItemSketchHelpers.sketchPartitions(hashingColumns, nUniques+1)_

        val sketchPerColumn =
          hashedTable
            .mapPartitions(sketching, preservesPartitioning = true)
            .treeReduce { case (columnAndSketchesA, columnAndSketchesB) =>
              columnAndSketchesA.map { case (column, sketchBinary) =>
                val sketchA = FreqSketch.apply(sketchBinary, LongType)
                  .asInstanceOf[LongFreqSketchImpl]
                val sketchB = FreqSketch.apply(columnAndSketchesB(column), LongType)
                  .asInstanceOf[LongFreqSketchImpl]
                sketchA.merge(sketchB)
                column -> sketchA.serializeTo()
              }
            }

        val hashingColumnsLookups = sketchPerColumn.map( kv => {
          val column = kv._1
          val sketchBinary = kv._2
          val sketch = FreqSketch.apply(sketchBinary, LongType).asInstanceOf[LongFreqSketchImpl]
          val items = sketch.getAllFrequentItems()
          val impl = sketch.impl.asInstanceOf[LongsSketch]
          val items_lookup = items.map(_._1.asInstanceOf[Long]).zipWithIndex.map { case (v, ind) =>
            (v, ind + 1)
          }.toMap
          (column -> items_lookup)
        })

        val numericColumns = outputIndexes.filter(kv =>
          selectedTypes(kv._1) == IntegerType || selectedTypes(kv._1) == DoubleType ).toMap
        val rangeLookups = numericColumns.map( col => {
          col._1 -> (0.0, globalMaxNumeric-0.0)
        })

        val nAllFeatures = selectedColumns.size
        val columnarData: RDD[(Long, Byte)] = hashedTable.zipWithIndex().flatMap (
           row => {
            val values = row._1
            val r = row._2
            val rindex = r * nAllFeatures
            val outputs = outputIndexes.map { col =>
              val column = col._1
              val cindex = col._2
              val dfIdx = cindex
              val dataType = selectedTypes(column)
              if (values.isNullAt(dfIdx)) {
                (rindex + cindex, 0.toByte)
              } else {
                val valueInColumn = (dataType: @switch) match {
                  case StringType  => {
                    val lookup = hashingColumnsLookups(column)
                    val hashIdx = lookup.getOrElse(values.getLong(dfIdx), 0)
                    hashIdx.toByte
                  }
                  case IntegerType  => {
                    val rawValue = values.getInt(dfIdx)
                    val mn = rangeLookups(column)._1
                    val range = rangeLookups(column)._2
                    val scaled = (rawValue - mn / range)*(nUniques-1)
                    (scaled+1).toByte
                  }
                  case DoubleType  => {
                    val rawValue = values.getDouble(dfIdx)
                    val mn = rangeLookups(column)._1
                    val range = rangeLookups(column)._2
                    val scaled = (rawValue - mn / range)*(nUniques-1)
                    (scaled+1).toByte
                  }
                }
                (rindex + cindex, valueInColumn)
              }
            }
            outputs
          }).sortByKey(numPartitions = 1) // put numPartitions parameter

        columnarData.persist(StorageLevel.MEMORY_AND_DISK_SER)
        columnarData.count()
        val duration = (System.nanoTime - t) / 1e9d
        println(f"prep x $duration")

        /*
        val t = System.nanoTime
        val stringSelectors = (1 to nVal).map(i => col(f"att$i").cast(StringType).alias
        (f"s_att$i")) :+ col("target")

        val df2 = df.select(stringSelectors: _*)

        val indexer = (1 to nVal).map(i => {
          val simple = new StringIndexerModel((1 to 100).map(i => f"$i").toArray)
          simple.setInputCol(f"s_att$i")
          simple.setOutputCol(f"idx_att$i")
          simple.setHandleInvalid("keep")
          simple
        }).toArray


        val assembler = new VectorAssembler()
          .setInputCols((1 to nVal).map(i => f"idx_att$i").toArray)
          .setOutputCol("features")

        val pipeline = new Pipeline().setStages(indexer :+ assembler)
        val output = pipeline.fit(df2).transform(df2)

        val output2 = output.select(col("target").alias("label"), col("features"))
        output2.cache()
        output2.count()
        val duration = (System.nanoTime - t) / 1e9d
        println(f"predp x $duration")

        val data = output2
          .rdd
          .map(row => LabeledPoint(row.getAs[Double]("label"), DenseVectorMLLib.fromML(row
            .getAs[DenseVector]("features"))))

      */

        val t0 = System.nanoTime
        MrmrSelector.trainColumnar(columnarData, 1, nAllFeatures)
        val duration0 = (System.nanoTime - t0) / 1e9d
        println(f"fastmrmr x $duration0")
      }

/*
    "calculate Mutual Information based on KLL sketches" in
      withSparkSession { sparkSession =>

        import org.apache.spark.sql.functions

        // create test dataset
        val nRows = 10000
        val nVal = 1000
        val nTargetBins = 100

        var df = sparkSession.read.format("parquet")
          .load(f"test-data/features_$nVal.parquet")
        df = df.withColumn("target",
            functions.round(functions.rand(10) * lit(nTargetBins)).cast(IntegerType))

        df.cache()
        df.count()
        // compute exac
        /*
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
         */

        val t0 = System.nanoTime

        // compute KLLs per target bucket (resultKllGrouped)
        val klls2 = (1 to nVal).map(i => stateful_kll_2(col(f"att$i"))
          .alias(f"kll_att$i"))
          .toArray
        var resultKllGrouped2 = df
          .agg(
            sum(when(col("target").isNotNull, 1)
              .otherwise(0)).alias("count_y"), klls2: _*)

        resultKllGrouped2.cache()
        resultKllGrouped2.count()
        val duration0 = (System.nanoTime - t0) / 1e9d
        println(f"kll x $duration0")

        val t1 = System.nanoTime

        // compute KLLs per target bucket (resultKllGrouped)
        val klls = (1 to nVal).map(i => stateful_kll_2(col(f"att$i"))
          .alias(f"kll_att$i"))
          .toArray
        var resultKllGrouped = df
          .groupBy("target")
          .agg(
            sum(when(col("target").isNotNull, 1)
              .otherwise(0)).alias("count_y"), klls: _*)

        resultKllGrouped.cache()
        resultKllGrouped.count()
        val duration = (System.nanoTime - t1) / 1e9d
        println(f"kll xy $duration")

        val t2 = System.nanoTime

        // aggregate KLLs across all buckets (resultKllAggregated)
        /*val kllAggs = (1 to nVal).map(i => stateful_kll_agg(Seq(
            col(f"kll_att$i").getField("sketch"),
            col(f"kll_att$i").getField("min"),
            col(f"kll_att$i").getField("max"),
            col(f"kll_att$i").getField("count")))
          .alias(f"pmf_agg_att$i")).toArray*/
        val kllAggs = (1 to nVal).map(i => stateful_kll_agg(col(f"kll_att$i"))
          .alias(f"agg_att$i")).toArray
        val resultKllAggregated = resultKllGrouped.agg(sum(col("count_y")).alias("total_y"),
            kllAggs: _*).first()

        // add target percentage to groped klls table
        resultKllGrouped = resultKllGrouped.withColumn("percent_y", col("count_y") /
            resultKllAggregated.getAs[Double]("total_y"))


        resultKllGrouped.cache()
        resultKllGrouped.count()
        val duration2 = (System.nanoTime - t2) / 1e9d
        println(f"kll x $duration2")

        val t2b = System.nanoTime

        // compute MI for all buckets (resultXYPMF)
        import org.apache.spark.sql.functions.udf
        def kllToKLDivergence(buckets: Int = 100, min: Option[Double] = None, max: Option[Double] =
        None, count: Option[Long] = None, pmfX: Array[Double])
        = udf(
          (kll: Array[Byte]) => {
            val state = KLLState.fromBytes(kll)
            val mn = min.getOrElse(state.globalMin)
            val mx = max.getOrElse(state.globalMax)
            val cnt = count.getOrElse(state.count)
            val pmfXY = state.qSketch.getPMF(buckets, mn, mx, cnt)
            val klDivergence = pmfXY.zipWithIndex.map {
              case (pX, i) => {
                if (pmfX(i) == 0 || pX.toDouble == 0) {
                  0
                } else {
                  pX * Math.log(pmfX(i) / pX.toDouble)
                }
              }
            }.sum * (-1)
            klDivergence
          })

        val selectsKLDivergence = (1 to nVal).map((i) => {
            val kllX = KLLState.fromBytes(resultKllAggregated.getAs[Array[Byte]](f"agg_att$i"))
            val minX = kllX.globalMin
            val maxX = kllX.globalMax
            val countX = kllX.count
            val pmfX = kllX.qSketch.getPMF(100, minX, maxX, countX)
            kllToKLDivergence(min = Some(minX), max = Some(maxX), pmfX = pmfX)(col(f"kll_att$i"))
              .as(f"pmf_agg_att$i")
          }) :+ col("count_y") :+ col("percent_y")

        // aggregate KLDivergence and sum/weight by bucket percentage to get mutual information
        val miAggs = (1 to nVal).map(i => sum(col(f"pmf_agg_att$i")*col("percent_y")).alias
        (f"mi_att$i")).toArray
        val result = resultKllGrouped.select(selectsKLDivergence: _*).agg(functions.count(col
        ("count_y")).alias("target_buckets"), miAggs: _*).first()

        println(result.getAs[Double]("mi_att1"))

        val duration2b = (System.nanoTime - t2b) / 1e9d
        println(f"mi on klls x $duration2b")
        val totalDuration = duration+duration2+duration2b
        println(f"mi on klls total $totalDuration")

        val t3 = System.nanoTime

        // compute correlations per target bucket (resultKllGrouped)
        val corrs = (1 to nVal).map(i => corr(col(f"att$i"),col(f"att$i"))
            .alias(f"kll_att$i"))
            .toArray
        var resultCorr = df
            .groupBy("target")
            .agg(
                sum(when(col("target").isNotNull, 1)
                  .otherwise(0)).alias("count_y"), corrs: _*)

        resultCorr.cache()
        resultCorr.count()
        val duration3 = (System.nanoTime - t3) / 1e9d
        println(f"correlations x $duration3")

        val t4 = System.nanoTime

        // compute KLLs per target bucket (resultKllGrouped)
        val miValues = (1 to nVal).map(i => MutualInformation(f"att$i", "target").calculate(df)
          .value)
        println(miValues(0))

        val duration4 = (System.nanoTime - t4) / 1e9d
        println(f"mi deequ x $duration4")


          // Define a UDF that wraps the upper Scala function defined above
        // You could also define the function in place, i.e. inside udf
        // but separating Scala functions from Spark SQL's UDFs allows for easier testing

        /*
        val (sketchXBin, minX, maxX, countX) = resultXY.map( row => {
          val X = row.getAs[Row]("att_pmf1")
          val sketch = X.getAs[Array[Byte]]("data")
          val min = X.getAs[Double]("minimum")
          val max = X.getAs[Double]("maximum")
          val count = X.getAs[Long]("count")
          (sketch, min, max, count)
        }: (Array[Byte], Double, Double, Long)).reduce((a, b) => {
          (KLLSketchSerializer.serializer.serialize(KLLSketchSerializer.serializer.deserialize(a._1)
            .merge(KLLSketchSerializer.serializer
            .deserialize(b._1))),
            Math.min(a._2, b._2),
            Math.max(a._3, b._3),
            a._4 + b._4
          )
        })
        val sketchX = KLLSketchSerializer.serializer.deserialize(sketchXBin)
        val distX = sketchX.getPMF(numBuckets, minX, maxX, countX)

        */
        /*
        val approxMI = resultXY.map( row => {
          val percentY = row.getAs[Double]("percent_y")
          val X = row.getAs[Row]("att_pmf1")
          val sketchXY = KLLSketchSerializer.serializer.deserialize(X.getAs[Array[Byte]]("sketch"))
          val countXY = X.getAs[Long]("count")
          val distXY = sketchXY.getPMF(numBuckets, minX, maxX, countXY)
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
        */
        // feature selection without/before transformations...

        // differentiate between numeric/categorial target
        // categorial target -> regular
        // categorial target > 500 distinct values -> regular
        // numerical target -> bucketing

        // differentiate between numeric/categorial feature
        // categorial features -> exact histogram
        // categorial feature > 500 distinct values -> exact histograms (exclude?)
        // numeric feature -> kll histogram
        // timestamp feature -> numeric
        // decimal feature -> numeric
        // date feature -> numeric
        // string -> exact histogram (adapted kl-diversion with lookup)

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
      */
  }
}
