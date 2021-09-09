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
import org.apache.spark.mllib.feature.MrmrSelector
import org.apache.spark.mllib.stat.{ExtendedMultivariateStatistics, ExtendedStatsConfig, ExtendedStatsHelperRow}
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
  private val numericColumns = schema.filter( kv => {
    kv.dataType match {
      case BooleanType | ByteType | ShortType | IntegerType | LongType | FloatType |
           DoubleType | TimestampType | DateType | DecimalType() => true
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
  private val otherColumns = schema.names.filterNot( name => numericColumns.contains(name) ).toSet

  def runFeatureSelection(df: DataFrame): Unit ={
    val dfLimited = df.limit(config.nRowLimit)
    dfLimited.persist(StorageLevel.MEMORY_AND_DISK_SER)

    val stats = ExtendedStatsHelperRow.computeColumnSummaryStatistics(dfLimited.rdd,
      schema.length,
      indexToType,
      ExtendedStatsConfig(maxFreqItems = config.nBuckets-2))

    val selectedColumns = preFilterColumns(stats)

    val subSelectedDf = dfLimited.select(selectedColumns.map(name => col(name)): _*)

    val transformedColumns = transformColumns(subSelectedDf.rdd, selectedColumns, stats)
    dfLimited.unpersist()

    val indexToFeatures = selectedColumns.zipWithIndex.map(kv => kv._2 -> kv._1).toMap

    MrmrSelector.trainColumnar(transformedColumns,
                               config.nSelectFeatures,
                               selectedColumns.length,
                               indexToFeatures)

  }

  private def transformColumns(rdd: RDD[Row], selectedColumns: Array[String],
                               stats: ExtendedMultivariateStatistics): RDD[(Long, Byte)] = {
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
        } else if (integralColumns.contains(name)) {
          if (stats.approxDistinct.get(statsIndex) > config.discretizationTreshold) {
            val rawValue = values.getInt(i)
            val mn = stats.min(statsIndex)
            val range = stats.max(statsIndex) - stats.min(statsIndex)
            val scaled = ((rawValue - mn) / range) * (config.nBuckets - 1) + 1
            byte = scaled.toByte
          } else {
            val hash = values.getInt(i).toDouble.toLong
            val lookup = stats.freqItems.get(statsIndex)
            byte = lookup.getOrElse(hash, 1.toByte)
          }
        } else if(numericColumns.contains(name)) {
          val rawValue = values.getDouble(i)
          val mn = stats.min(statsIndex)
          val range = stats.max(statsIndex) - stats.min(statsIndex)
          val scaled = ((rawValue - mn) / range) * (config.nBuckets - 1) + 1
          byte = scaled.toByte
        } else {
          val hash = stringHash(values.getString(i), 42)
          val lookup = stats.freqItems.get(statsIndex)
          byte = lookup.getOrElse(hash, 1.toByte)
        }
        (index, byte)
      }
      inputs
    })
    columnarData
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
      if (numericColumns.contains(name)) {
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
        if (noFrequentItems) reasons += f"less than two frequent items"
        println(f"Column $name was pre-filtered, because ${reasons.mkString(", ")}")
      }

      filter
    }).keys.toArray :+ config.target

    selectedColumns
  }

}
