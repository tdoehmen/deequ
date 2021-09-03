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
import com.amazon.deequ.analyzers.KLLParameters
import com.amazon.deequ.profiles.{ColumnProfiler, ColumnProfiles}
import com.amazon.deequ.utils.FixtureSupport
import org.apache.datasketches.frequencies.LongsSketch
import org.apache.spark.mllib.feature.MrmrSelector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.expressions.XxHash64Function
import org.apache.spark.sql.catalyst.expressions.aggregate.{FreqSketch, FrequentItemSketchHelpers, LongFreqSketchImpl}
import org.apache.spark.sql.functions.{min, _}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Row, functions}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.unsafe.types.UTF8String
import org.scalatest.{Matchers, WordSpec}

import scala.collection.mutable.ArrayBuffer

class TestFeatureSelection extends WordSpec with Matchers with SparkContextSpec
  with FixtureSupport {

  "Approximate Mutual Information" should {

    "calculate MRMR with fastmrmr" in
      withSparkSession { sparkSession =>
        /*
        Evaluation
    1000 cols (500 str/num), 100 target bins, 1 feature selected, 10k rows -> 45s->112s, 29s
    1000 cols (500 str/num), 100 target bins, 100 feature selected, 10k rows -> 45s->112s, 2229s
    1000 cols (500 str/num), 100 target bins, 1 feature selected, 50k rows -> 261s->488s, 145s
    1000 cols (500 str/num), 100 target bins, 1 feature selected, 100k rows ->
         */

        // create test dataset
        val nBuckets = 255 // current fastmrmr impl allows max 255 (byte-size)
        val nSelectFeatures = 10
        val nRowLimit = 1000000
        val normalizedVarianceThreshold = 0.01
        val distinctnessThresholdIntegral = 0.9
        val distinctnessThresholdOther = 0.5
        val completenessThreshold = 0.5
        val discretizationTreshold = 100
        val frequentItemSketchSize = 1024
        val target = "target"

        val byteCompatibleTypes = Seq(BooleanType, ByteType)
        val numericTypes = Seq(ShortType, IntegerType, LongType, FloatType, DoubleType,
          DecimalType, TimestampType, DateType)
        val integralTypes = Seq(ShortType, IntegerType, LongType, TimestampType, DateType)

        val tLoadingAndStats = System.nanoTime

        val targetInp = "Survived"
        var df = sparkSession.read.format("csv")
          .option("inferSchema", "true")
          .option("header", "true")
          .load("test-data/titanic.csv")
          .withColumnRenamed(targetInp, target)
        /*

        val nVal = 1000
        val nTargetBins = 100
        var df = sparkSession.read.format("parquet")
          .load(f"test-data/features_int_50k_$nVal.parquet")
        df = df.withColumn(target,
          functions.round(functions.rand(10) * lit(nTargetBins)))
*/
        df.limit(nRowLimit)
        df.persist(StorageLevel.MEMORY_AND_DISK_SER)

        val nPart = df.rdd.getNumPartitions
        println("number of partitions")
        println(nPart)


        /*val tdeequ = System.nanoTime
        // stats for feature selection and binning (usually provided by profiling json)
        val profile = ColumnProfiler.profileOptimized(df, correlation = false)
        val stats = profile.profiles
        val durationdeequ = (System.nanoTime - tdeequ) / 1e9d
        println(f"stats deequ filter x $durationdeequ")
         */

        // stats for feature selection and binning (a bit faster than deequ profiling (5-50%)
        val numStatsAggs = df.schema.filter(c => numericTypes.contains(c.dataType))
          .flatMap(c => Seq(
          min(col(c.name)).cast(DoubleType).alias(c.name+"_min"),
          max(col(c.name)).cast(DoubleType).alias(c.name+"_max"),
          approx_count_distinct(col(c.name)).cast(DoubleType).alias(c.name+"_dist"),
          count(when(col(c.name).isNull || col(c.name).isNaN, lit(1)))
              .cast(DoubleType).alias(c.name + "_count_null"),
          stddev(col(c.name).cast(DoubleType)).cast(DoubleType).alias(c.name+"_stddev"),
          mean(col(c.name).cast(DoubleType)).cast(DoubleType).alias(c.name+"_mean")))
        val otherStatsAggs = df.schema.filterNot(c => numericTypes.contains(c.dataType))
          .flatMap (c => Seq(
          approx_count_distinct(col(c.name)).cast(DoubleType).alias(c.name+"_dist"),
          count(when(col(c.name).isNull, lit(1))).cast(DoubleType).alias(c.name + "_count_null")))
        val generalCount = Seq(count(lit(1)).cast(DoubleType).alias("_count"))
        val stats = df.select(numStatsAggs ++ otherStatsAggs ++ generalCount: _*).first()

        //println(stats)

        // select features (simple low variance, low completeness, high distinctness filter)
        val selectedColumns = df.schema.filter(kv => kv.name != target).filterNot(c => {
          val distinct = stats.getAs[Double](c.name+"_dist")
          val count = stats.getAs[Double]("_count")
          val countNull = stats.getAs[Double](c.name+"_count_null")
          val countSafe = if (count == 0) 1 else count
          val noVariance = distinct == 1
          val lowCompleteness = (count - countNull) / countSafe < completenessThreshold
          if (numericTypes.contains(c.dataType)) {
            val stddev = stats.getAs[Double](c.name+"_stddev")
            val mean = stats.getAs[Double](c.name+"_mean")
            val meanSafe = if (mean==0) 1e-100 else mean
            val lowVariance = (stddev/meanSafe)*(stddev/meanSafe) < normalizedVarianceThreshold
            if (integralTypes.contains(c.dataType)) {
              val highDistinctness = (distinct / countSafe) > distinctnessThresholdIntegral
              noVariance || lowCompleteness || lowVariance || highDistinctness
            } else {
              noVariance || lowCompleteness || lowVariance
            }
          } else {
            val highDistinctness = (distinct / countSafe) > distinctnessThresholdOther
            noVariance || lowCompleteness || highDistinctness
          }
        }).map(c => c.name) :+ target

        df = df.select(selectedColumns.map(c => col(c)).toArray: _*)

        val durationLoadingAndStats = (System.nanoTime - tLoadingAndStats) / 1e9d
        println(f"loading, stats, pre-filter x $durationLoadingAndStats")

        val tHashingBucketing = System.nanoTime

        //val selectedColumns = (1 to nVal).map(i => f"att$i").toArray :+ "target"
        val selectedIndexes = df.schema.fields
          .map { _.name }
          .zipWithIndex
          .toMap
        val selectedTypes = df.schema
          .map { info => info.name -> info.dataType }
          .toMap

        println("selected features and indexes")
        println(selectedIndexes.map(i => (i._1, i._2 + 1)).toMap)

        // create bucket lookups
        val bucketingLookupsBuf = ArrayBuffer[(String, (Double, Double))]()
        selectedTypes.filter(kv => numericTypes.contains(kv._2)).foreach(c => {
          val min = stats.getAs[Double](c._1+"_min")
          val max = stats.getAs[Double](c._1+"_max")
          val dist = stats.getAs[Double](c._1+"_dist")
          if (dist > discretizationTreshold) {
            bucketingLookupsBuf.append((c._1, (min, max-min)))
          }
        })
        val bucketingLookups = bucketingLookupsBuf.toMap

        println("bucketing lookups")
        println(bucketingLookups)

        // all columns which are neither bucketized, nor byte compatible are being hashed
        val hashingColumns = selectedIndexes.filterNot(kv => {
          bucketingLookups.contains(kv._1) || byteCompatibleTypes.contains(selectedTypes(kv._1))
        })

        val bucketizeFn = (rawValue: Double, column: String) => {
          val mn = bucketingLookups(column)._1
          val range = bucketingLookups(column)._2
          val scaled = ((rawValue - mn) / range) * (nBuckets - 1)
          scaled.toByte
        }

        // binarize, bucketize+binarize, or create hashes
        val nAllFeatures = selectedIndexes.size
        val hashedTable: RDD[Row] = df.rdd.map (
          row => {
            val values = row
            val outputs = selectedColumns.map { column =>
              val dfIdx = selectedIndexes(column)
              val dataType = selectedTypes(column)
              if (values.isNullAt(dfIdx)) {
                0
              } else if (byteCompatibleTypes.contains(dataType)) {
                dataType match {
                  case BooleanType => if (values.getBoolean(dfIdx)) 1.toByte else 0.toByte
                  case ByteType => values.getByte(dfIdx)
                  case _ => // Not supported
                    throw new IllegalArgumentException(f"$dataType not byte compatible")
                }
              } else if (bucketingLookups.contains(column)) {
                dataType match {
                  case ShortType => bucketizeFn(values.getShort(dfIdx).toDouble, column)
                  case IntegerType =>  bucketizeFn(values.getInt(dfIdx).toDouble, column)
                  case LongType =>  bucketizeFn(values.getLong(dfIdx).toDouble, column)
                  case FloatType =>  bucketizeFn(values.getFloat(dfIdx).toDouble, column)
                  case DoubleType => bucketizeFn(values.getDouble(dfIdx), column)
                  case DecimalType() => bucketizeFn(values.getDecimal(dfIdx).doubleValue(), column)
                  case TimestampType => bucketizeFn(values.getTimestamp(dfIdx).getTime, column)
                  case DateType => bucketizeFn(values.getDate(dfIdx).getTime, column)
                  case _ => // Not supported
                    throw new IllegalArgumentException(f"$dataType not supported for bucketing")
                }
              } else {
                dataType match {
                  case StringType =>
                    XxHash64Function.hash(UTF8String.fromString(values.getString(dfIdx)),
                      dataType, 42L)
                  case _ => // Numerics, Binary, Array, Map, Struct
                    XxHash64Function.hash(values.get(dfIdx), dataType, 42L)
                }
              }
            }
            Row(outputs: _*)
          })

        hashedTable.persist(StorageLevel.MEMORY_AND_DISK_SER)
        hashedTable.count()
        df.unpersist()
        val durationHashingBucketing = (System.nanoTime - tHashingBucketing) / 1e9d
        println(f"hashing, bucketing x $durationHashingBucketing")

        val tFrequentItemSketches = System.nanoTime

        val sketching = FrequentItemSketchHelpers.sketchPartitions(hashingColumns,
          frequentItemSketchSize) _
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
          val sketchBinary = kv._2
          val sketch = FreqSketch.apply(sketchBinary, LongType).asInstanceOf[LongFreqSketchImpl]
          val items = sketch.getAllFrequentItems()
          val impl = sketch.impl.asInstanceOf[LongsSketch]
          val items_lookup = items.take(Math.min(items.length, nBuckets))
            .map(_._1.asInstanceOf[Long])
            .zipWithIndex
            .map {
              case
                (v, ind) =>
                (v, ind + 1)
            }.toMap
          (kv._1 -> items_lookup)
        })


        println("hashing colums")
        println(hashingColumnsLookups.keys)

        val durationFrequentItemSketches = (System.nanoTime - tFrequentItemSketches) / 1e9d
        println(f"frequent item sketches x $durationFrequentItemSketches")

        val tColumnarBinary = System.nanoTime

        val columnarData: RDD[(Long, Byte)] = hashedTable.zipWithIndex().flatMap (
           row => {
            val values = row._1
            val r = row._2
            val rindex = r * nAllFeatures
            val outputs = selectedIndexes.map { c =>
              val column = c._1
              val dfIdx = c._2
              if (values.isNullAt(dfIdx)) {
                (rindex + dfIdx, 0.toByte)
              } else {
                val valueInColumn = if (hashingColumns.contains(column)) {
                  val lookup = hashingColumnsLookups(column)
                  val hashIdx = lookup.getOrElse(values.getLong(dfIdx), 0)
                  hashIdx.toByte
                } else {
                  values.getLong(dfIdx).toByte
                }
                (rindex + dfIdx, valueInColumn)
              }
            }
            outputs
          }).sortByKey(numPartitions = nPart) // put numPartitions parameter

        columnarData.persist(StorageLevel.MEMORY_AND_DISK_SER)
        columnarData.count()
        hashedTable.unpersist()

        val durationColumnarBinary = (System.nanoTime - tColumnarBinary) / 1e9d
        println(f"columnar binary rdd construction x $durationColumnarBinary")

        val durationTotal = durationLoadingAndStats + durationHashingBucketing +
          durationFrequentItemSketches + durationColumnarBinary
        println(f"prep total x $durationTotal")

        val tSelection = System.nanoTime

        MrmrSelector.trainColumnar(columnarData,  nSelectFeatures,  nAllFeatures)

        val durationSelection = (System.nanoTime - tSelection) / 1e9d
        println(f"fastmrmr x $durationSelection")
      }
  }
}
