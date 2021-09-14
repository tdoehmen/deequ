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
import com.amazon.deequ.utils.FixtureSupport
import org.apache.spark.sql.functions.{count, mean, min, _}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Row, functions}
import org.apache.spark.storage.StorageLevel
import org.scalatest.{Matchers, WordSpec}

class FeatureSelectionBenchmark extends WordSpec with Matchers with SparkContextSpec
  with FixtureSupport {

  "Feature Selection Benchmark" should {

    "DataFrame-based stats equal ExtendedMultivariateOnlineSummarize stats" in
      withSparkSession { sparkSession =>

        val nVal = 100
        val df = sparkSession.read.format("parquet").load(f"test-data/features_int_10k_$nVal" +
          f".parquet")
        df.persist(StorageLevel.MEMORY_AND_DISK_SER)
        df.count()

        val tStatsRdd = System.nanoTime


        val indexToType = df.schema.zipWithIndex.map(kv => kv._2 -> kv._1.dataType).toMap
        val statsrow = ExtendedStatsHelper.computeColumnSummaryStatistics(df.rdd,
          df.schema.length,
          indexToType,
          ExtendedStatsConfig(frequentItems = false, approxQuantiles = true))

        println(statsrow.min.mkString(" "))
        println(statsrow.max.mkString(" "))
        println(statsrow.approxDistinct.get.mkString(" "))

        val durationStatsRdd = (System.nanoTime - tStatsRdd) / 1e9d
        println(f"summary stats rdd x $durationStatsRdd")


        val tStats = System.nanoTime

        val numericColumns = df.schema.filter( kv => {
          kv.dataType match {
            case BooleanType | ByteType | ShortType | IntegerType | LongType | FloatType |
                 DoubleType | TimestampType | DateType | DecimalType() => true
            case _ =>
              false
          }
        })

        val otherColumns = df.schema.filterNot( kv => {
          kv.dataType match {
            case BooleanType | ByteType | ShortType | IntegerType | LongType | FloatType |
                 DoubleType | TimestampType | DateType | DecimalType() => true
            case _ =>
              false
          }
        })

        val numStatsAggs = numericColumns.flatMap(c => {
          val withoutNullAndNan = when(!col(c.name).isNull && !col(c.name).isNaN, col(c.name))
          Seq(
            min(withoutNullAndNan).cast(DoubleType).alias(c.name+"_min"),
            max(withoutNullAndNan).cast(DoubleType).alias(c.name+"_max"),
            approx_count_distinct(col(c.name)).alias(c.name+"_dist"),
            count(when(col(c.name).isNull || col(c.name).isNaN, lit(1)))
              .alias(c.name + "_count_null"),
            stddev(withoutNullAndNan).alias(c.name+"_stddev"),
            mean(withoutNullAndNan).alias(c.name+"_mean"))
        })
        val otherStatsAggs = otherColumns
          .flatMap (c => Seq(
            approx_count_distinct(col(c.name)).alias(c.name+"_dist"),
            count(when(col(c.name).isNull, lit(1))).alias(c.name + "_count_null")))
        val generalCount = Seq(count(lit(1)).alias("_count"))

        val stats = df.select(numStatsAggs ++ otherStatsAggs ++
          generalCount: _*)
        stats.show()


        val durationStats = (System.nanoTime - tStats) / 1e9d
        println(f"summary stats x $durationStats")

      }
  }
}
