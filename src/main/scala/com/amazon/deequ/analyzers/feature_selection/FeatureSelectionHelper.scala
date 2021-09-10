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

package com.amazon.deequ.analyzers.feature_selection

import org.apache.datasketches.frequencies.LongsSketch
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.feature.MrmrSelector
import org.apache.spark.mllib.stat.{ExtendedMultivariateStatistics, ExtendedStatsConfig, ExtendedStatsHelper, NumerizationHelper}
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.types.{BooleanType, ByteType, DataType, DateType, DecimalType, DoubleType, FloatType, IntegerType, LongType, ShortType, StructType, TimestampType}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.col
import org.apache.spark.storage.StorageLevel

import scala.collection.mutable.ArrayBuffer
import scala.util.hashing.MurmurHash3.stringHash


case class FeatureSelectionConfig(target: String = "target",
                                  nSelectFeatures: Int = -1,
                                  nBuckets: Int = 255,
                                  nRowLimit: Int = 1000000,
                                  normalizedVarianceThreshold: Double = 0.01,
                                  distinctnessThresholdIntegral: Double = 0.9,
                                  distinctnessThresholdOther: Double = 0.5,
                                  completenessThreshold: Double = 0.5,
                                  discretizationTreshold: Int = 100,
                                  frequentItemSketchSize: Int = 1024
                                 )


class FeatureSelectionHelper(schema: StructType, config: FeatureSelectionConfig =
  FeatureSelectionConfig()) extends Serializable {

  private val indexToType = schema.zipWithIndex.map(kv => kv._2 -> kv._1.dataType).toMap
  private val nameToIndex = schema.zipWithIndex.map(kv => kv._1.name -> kv._2).toMap
  private val nameToType = schema.map(kv => kv.name -> kv.dataType).toMap
  private val fractionalColumns = schema.filter( kv => {
    kv.dataType match {
      case FloatType | DoubleType | DecimalType() => true
      case _ =>
        false
    }
  }).map(kv => kv.name).toSet
  private val integralColumns = schema.filter( kv => {
    kv.dataType match {
      case BooleanType | ByteType | ShortType | IntegerType | LongType |
           TimestampType | DateType => true
      case _ =>
        false
    }
  }).map(kv => kv.name).toSet

  def runFeatureSelection(df: DataFrame): Unit ={
    val dfLimited = df.limit(config.nRowLimit)
    dfLimited.persist(StorageLevel.MEMORY_AND_DISK_SER)

    val tStart = System.nanoTime

    val stats = ExtendedStatsHelper.computeColumnSummaryStatistics(dfLimited.rdd,
      schema.length,
      indexToType,
      ExtendedStatsConfig(maxFreqItems = config.nBuckets-2))

    //println(stats.min.toArray.mkString(" "))
    //println(stats.max.toArray.mkString(" "))
    //println(stats.freqItems.get.map(seq => seq.mkString(", ")).mkString("\n"))

    val selectedColumns = preFilterColumns(stats)

    val subSelectedDf = dfLimited.select(selectedColumns.map(name => col(name)): _*)

    //val sc = subSelectedDf.rdd.context
    //val bStats = sc.broadcast(stats)

    val transformedColumns = transformColumns(subSelectedDf.rdd, selectedColumns, stats)
    dfLimited.unpersist()
    transformedColumns.persist(StorageLevel.MEMORY_AND_DISK_SER)
    transformedColumns.count()

    val durationPreprocess = (System.nanoTime - tStart) / 1e9d
    println(f"preprocess data x $durationPreprocess")

    val indexToFeatures = selectedColumns.zipWithIndex.map(kv => kv._2 -> kv._1).toMap

    MrmrSelector.trainColumnar(transformedColumns,
                               config.nSelectFeatures,
                               selectedColumns.length,
                               indexToFeatures)

  }

  // feature pre-filter (except target)
  private def preFilterColumns(stats: ExtendedMultivariateStatistics)
  : Array[String] = {
    val selectedColumns = nameToIndex.filter(kv => kv._1 != config.target).filterNot(kv => {
      val name = kv._1
      val index = kv._2
      val distinct = stats.approxDistinct.get(index)
      val count = stats.count
      val countNull = stats.count - stats.numNonzeros(index)
      val countSafe = if (count == 0) 1 else count
      val noVariance = distinct == 1
      val completeness = (count - countNull) / countSafe
      val lowCompleteness = completeness < config.completenessThreshold
      var lowVariance = false
      val distinctness = (distinct / countSafe)
      var highDistinctnessIntegral = false
      var highDistinctness = false
      var noFrequentItems = false
      var normVariance = 0.0
      if (fractionalColumns.contains(name) || integralColumns.contains(name)) {
        val stddev = stats.variance(index) * stats.variance(index)
        val mean = stats.mean(index)
        val meanSafe = if (mean == 0) 1e-100 else mean
        normVariance = (stddev / meanSafe) * (stddev / meanSafe)
        lowVariance = normVariance < config.normalizedVarianceThreshold
        if (integralColumns.contains(name)) {
          highDistinctnessIntegral = distinctness > config.distinctnessThresholdIntegral
        }
      } else {
        highDistinctness = distinctness > config.distinctnessThresholdOther
        noFrequentItems = stats.freqItems.get(index).size < 2
      }
      val filter = noVariance || lowCompleteness || lowVariance || highDistinctness |
        highDistinctnessIntegral || noFrequentItems

      if (filter) {
        val reasons = ArrayBuffer[String]()
        if (noVariance) reasons += f"variance is 0"
        if (lowCompleteness) reasons += f"completeness $completeness < ${
          config.completenessThreshold  }"
        if (lowVariance) reasons += f"variance $normVariance < ${
          config.normalizedVarianceThreshold }"
        if (highDistinctnessIntegral) reasons += f"integral distinctness $distinctness > ${
          config.distinctnessThresholdIntegral }"
        if (highDistinctness) reasons += f"distinctness $distinctness > ${
          config.distinctnessThresholdOther }"
        if (noFrequentItems) reasons += f"less than two most frequent items"
        println(f"Column $name was pre-filtered, because ${reasons.mkString(", ")}")
      }

      filter
    }).keys.toArray :+ config.target

    selectedColumns
  }


  private def transformColumns(rdd: RDD[Row], selectedColumns: Array[String],
                               stats: ExtendedMultivariateStatistics): RDD[(Long, Byte)]
  = {
    val nAllFeatures = selectedColumns.length

    val columnarData: RDD[(Long, Byte)] = rdd.zipWithIndex().flatMap ({ kv =>
      val values = kv._1
      val r = kv._2
      val rindex = r * nAllFeatures
      val inputs = for(i <- 0 until nAllFeatures) yield {
        val name = selectedColumns(i)
        val statsIndex = nameToIndex(name)
        val index = rindex + i
        var byte = 0.toByte
        if (values.isNullAt(i)) {
          byte = 0.toByte
        } else {
          val value = NumerizationHelper.numerize(values, i, nameToType(name))
          if (Seq(Double.PositiveInfinity, Double.NegativeInfinity, Double.NaN).contains(value)) {
            byte = 0.toByte
          } else if(fractionalColumns.contains(name) || (integralColumns.contains(name) &&
            stats.approxDistinct.get(statsIndex) > config.discretizationTreshold )) {
            val rawValue = value
            val mn = stats.min(statsIndex)
            val range = stats.max(statsIndex) - stats.min(statsIndex)
            val scaled = ((rawValue - mn) / range) * (config.nBuckets - 1) + 1
            byte = scaled.toByte
          } else {
            val lookup = stats.freqItems.get(statsIndex)
            // hash-lookups-> 0-253; miss: -1 => range (-1-253)+2 = 1-255 (leaves 0-index for nulls)
            byte = (lookup.getOrElse(value.toLong, -1) + 2).toByte
          }
        }
        (index, byte)
      }
      inputs
    })
    columnarData
  }


}
