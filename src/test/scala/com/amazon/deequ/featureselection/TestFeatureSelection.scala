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

import java.util.Calendar

import com.amazon.deequ.SparkContextSpec
import com.amazon.deequ.analyzers.{ApproxCountDistinct, KLLParameters}
import com.amazon.deequ.profiles.{ColumnProfiler, ColumnProfiles}
import com.amazon.deequ.utils.FixtureSupport
import org.apache.datasketches.frequencies.LongsSketch
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.mllib.linalg.{DenseVector => DenseVectorMLLib}
import org.apache.spark.mllib.feature.{MrmrSelector, StandardScalerModel}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.stat.{ExtendedStatsConfig, ExtendedStatsHelper, MultivariateOnlineSummarizer, MultivariateStatisticalSummary, Statistics}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.expressions.XxHash64Function
import org.apache.spark.sql.catalyst.expressions.aggregate.{FreqSketch, FrequentItemSketchHelpers, LongFreqSketchImpl, StatefulHyperloglogPlus}
import org.apache.spark.sql.functions.{count, mean, min, _}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Row, functions}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.unsafe.types.UTF8String
import org.scalatest.{Matchers, WordSpec}

import scala.collection.immutable.ListMap
import scala.collection.mutable.ArrayBuffer

class TestFeatureSelection extends WordSpec with Matchers with SparkContextSpec
  with FixtureSupport {

  "Approximate Mutual Information" should {

    "stats runtime" in
      withSparkSession { sparkSession =>

        val nRows = 10000
        val nTargetBins = 100
        val nVal = 1000
        val df = sparkSession.read.format("parquet").load(f"test-data/features_int_10k_$nVal" +
          f".parquet")
        df.persist(StorageLevel.MEMORY_AND_DISK_SER)
        df.count()

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


        val hashs = otherColumns.map( kv => {
          xxhash64(col(kv.name)).alias(kv.name)
        })

        val doubles = numericColumns.map(kv => {
          if (kv.dataType == FloatType) {
            when(col(kv.name).isin(Float.PositiveInfinity, Float.NegativeInfinity),
              Float.NaN).otherwise(col(kv.name)).cast(DoubleType).alias(kv.name)
          } else if (kv.dataType == DoubleType) {
            when(col(kv.name).isin(Double.PositiveInfinity, Double.NegativeInfinity),
              Double.NaN).otherwise(col(kv.name)).alias(kv.name)
          } else if (kv.dataType == DateType) {
            to_timestamp(col(kv.name)).cast(DoubleType)
          } else if (kv.dataType == CalendarIntervalType) {
            to_timestamp(col(kv.name)).cast(DoubleType)
          } else {
            col(kv.name).cast(DoubleType).alias(kv.name)
          }
        })

        val tc = System.nanoTime
        val dfc = df.select(hashs ++ doubles: _*).na.fill(0.0, numericColumns.map(kv => kv.name))
        dfc.persist(StorageLevel.MEMORY_AND_DISK_SER)
        dfc.count()
        val durationc = (System.nanoTime - tc) / 1e9d
        println(f"data conversion x $durationc")


        val tv = System.nanoTime

        val assembler = new VectorAssembler().setHandleInvalid("skip")
          .setInputCols(dfc.schema.map(kv => kv.name).toArray)
          .setOutputCol("features")
        val output = assembler.transform(dfc)
        val data = output.select("features")
          .rdd
          .map(row => org.apache.spark.mllib.linalg.Vectors.fromML(
            row.getAs[org.apache.spark.ml.linalg.DenseVector]("features")))

        val durationv = (System.nanoTime - tv) / 1e9d
        println(f"vectorization x $durationv")


        val tvs = System.nanoTime
        val summary = ExtendedStatsHelper.computeColumnSummaryStatistics(data, dfc.schema.length,
          ExtendedStatsConfig(frequentItems = false))
        val durationvs = (System.nanoTime - tvs) / 1e9d
        println(f"stats vectorized x $durationvs")
        dfc.unpersist()

        println(summary.approxDistinct.get)
        //println(summary.freqItems.get(0).mkString("Array(", ", ", ")"))

        // stats for feature selection and binning (a bit faster than deequ profiling (5-50%)
        // but still slow. Should be changed to rdd-based aggregations!!!
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

        val tdf = System.nanoTime
        val stats = df.select(numStatsAggs ++ otherStatsAggs ++
          generalCount: _*).first()
        val durationdf = (System.nanoTime - tdf) / 1e9d
        println(f"stats dataframe x $durationdf")


      }

    "rdd-based statistics and data prep2" in
      withSparkSession { sparkSession =>
        import sparkSession.implicits._
        var df =
          Seq(
            (BigDecimal(1.0), BigDecimal(0.0)),
            (BigDecimal(2.0), BigDecimal(0.0)),
            (BigDecimal(3.0), BigDecimal(0.0)),
            (BigDecimal(4.0), BigDecimal(5.0)),
            (BigDecimal(5.0), BigDecimal(6.0)),
            (BigDecimal(6.0), BigDecimal(7.0))
          ).toDF("att1", "att2")

        val result = ApproxCountDistinct("att1").calculate(df).value
        println(result)
        val types = df.schema.zipWithIndex.map(kv => kv._2 -> kv._1.dataType).toMap
        val columnarData: RDD[(Long, Array[Long])] = df.rdd.zipWithIndex().map(
          row => {
            val values = row._1
            val r = row._2
            val outputs = (0 to values.length).map(i => {
              println(types(i))
              println(values.get(i).getClass)
              XxHash64Function.hash(values.get(i), types(i), 42L)
            }).toArray
            (r, outputs)
          }
        )

        println(columnarData.collect())

      }


    "df-based hashing" in
      withSparkSession { sparkSession =>

        val nRows = 10000
        val nTargetBins = 100
        val nVal = 1000
        val df = sparkSession.read.format("parquet").load(f"test-data/features_int_10k_$nVal" +
          f".parquet")
        df.persist(StorageLevel.MEMORY_AND_DISK_SER)
        df.count()

        val hash = df.schema.map( kv => {
          xxhash64(col(kv.name)).alias(kv.name)
        })

        val t0 = System.nanoTime
        val dfh = df.select(hash: _*)
        dfh.persist(StorageLevel.MEMORY_AND_DISK_SER)
        val duration0 = (System.nanoTime - t0) / 1e9d
        println(f"hashing x $duration0")

        val doubles = df.schema.filter( kv => {
          kv.dataType match {
            case BooleanType | ByteType | ShortType | IntegerType | LongType | FloatType |
                DoubleType | TimestampType | DateType | DecimalType() => true
            case _ =>
              false
          }
        }).map(kv => {
          if (kv.dataType == FloatType) {
            when(col(kv.name).isin(Float.PositiveInfinity, Float.NegativeInfinity),
              Float.NaN).otherwise(col(kv.name)).cast(DoubleType).alias(kv.name)
          } else if (kv.dataType == DoubleType) {
            when(col(kv.name).isin(Double.PositiveInfinity, Double.NegativeInfinity),
              Double.NaN).otherwise(col(kv.name)).alias(kv.name)
          } else if (kv.dataType == DateType) {
            to_timestamp(col(kv.name)).cast(DoubleType)
          } else if (kv.dataType == CalendarIntervalType) {
            to_timestamp(col(kv.name)).cast(DoubleType)
          } else {
            col(kv.name).cast(DoubleType).alias(kv.name)
          }
        })

        val t0d = System.nanoTime
        var dfd = df.select(doubles: _*)
        dfd = dfd.na.fill(0.0, dfd.schema.map(kv => kv.name))
        val assembler = new VectorAssembler().setHandleInvalid("skip")
          .setInputCols(dfd.schema.map(kv => kv.name).toArray)
          .setOutputCol("features")


        val output = assembler.transform(dfd)
        val data = output.select("features")
          .rdd
          .map(row => org.apache.spark.mllib.linalg.Vectors.fromML(
            row.getAs[org.apache.spark.ml.linalg.DenseVector]("features")))

        output.select("features").show()
        val summary = ExtendedStatsHelper.computeColumnSummaryStatistics(data, dfd.schema.length)

        /*val summary: MultivariateStatisticalSummary = Statistics.colStats(data)
        println(summary.min)  // a dense vector containing the mean value for each column
        println(summary.max)  // a dense vector containing the mean value for each column
        println(summary.mean)  // a dense vector containing the mean value for each column
        println(summary.variance)  // column-wise variance
        println(summary.numNonzeros)  // number of nonzeros in each column*/

        //dfd.summary("min", "max", "stddev").collect()
        val duration0d= (System.nanoTime - t0d) / 1e9d
        println(f"doubles x $duration0d")

        println(summary.approxDistinct.get)
        println(summary.freqItems.get(0).mkString("Array(", ", ", ")"))

        val t0s = System.nanoTime
        val scaler3 = new StandardScalerModel(org.apache.spark.mllib.linalg.Vectors.dense(summary
          .variance.toArray.map(v=>Math.sqrt(v))),
          summary.mean)
        scaler3.transform(data).collect()
          //.foreach(v => println(v.toArray.mkString("Array(", ", ",")")))
        val duration0s= (System.nanoTime - t0s) / 1e9d
        println(f"sclaing x $duration0s")
      }

    "rdd-based statistics and data prep" in
      withSparkSession { sparkSession =>
        import sparkSession.implicits._
        var df =
          Seq(
            (BigDecimal(1.0), BigDecimal(0.0)),
            (BigDecimal(2.0), BigDecimal(0.0)),
            (BigDecimal(3.0), BigDecimal(0.0)),
            (BigDecimal(4.0), BigDecimal(5.0)),
            (BigDecimal(5.0), BigDecimal(6.0)),
            (BigDecimal(6.0), BigDecimal(7.0))
          ).toDF("att1", "att2")

        val result = ApproxCountDistinct("att1").calculate(df).value
        println(result)
        val types = df.schema.zipWithIndex.map(kv => kv._2 -> kv._1.dataType).toMap
        val columnarData: RDD[(Long, Array[Long])] = df.rdd.zipWithIndex().map(
          row => {
            val values = row._1
            val r = row._2
            val outputs = (0 to values.length).map(i => {
              println(types(i))
              println(values.get(i).getClass)
              XxHash64Function.hash(values.get(i), types(i), 42L)
            }).toArray
            (r, outputs)
          }
        )

        println(columnarData.collect())

      }

    "calculate MRMR with fastmrmr oldschool" in
      withSparkSession { sparkSession =>
        // create test dataset
        val nRows = 10000
        val nVal = 5000
        val nTargetBins = 100

        var df = sparkSession.read.format("parquet")
          .load(f"test-data/features_int_1k_$nVal.parquet")
        df = df.withColumn("target",
          functions.round(functions.rand(10) * lit(nTargetBins)))

        println(df.rdd.getNumPartitions)
        df.cache()
        df.count()

        val assembler = new VectorAssembler()
          .setInputCols((1 to nVal).map(i => f"att$i").toArray)
          .setOutputCol("features")

        val output = assembler.transform(df)

        val data = output.select(col("target").alias("label"), col("features"))
          .rdd
          .map(row => LabeledPoint(row.getAs[Double]("label"), DenseVectorMLLib.fromML(row
            .getAs[DenseVector]("features"))))


        val t0 = System.nanoTime
        MrmrSelector.train(data, 1, 1)
        val duration0 = (System.nanoTime - t0) / 1e9d
        println(f"fastmrmr x $duration0")
      }

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
        val nSelectFeatures = -1
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
        val schema = StructType(
            StructField("bool", BooleanType, true) ::
            StructField("byte", ByteType, true) ::
            StructField("short", ShortType, true) ::
            StructField("int", IntegerType, true) ::
            StructField("long", LongType, true) ::
            StructField("float", FloatType, true) ::
            StructField("dbl", DoubleType, true) ::
            StructField("dec", DecimalType(38,28), true) ::
            StructField("ts", TimestampType, true) ::
            StructField("dt", DateType, true) ::
            StructField("str", StringType, true) ::
            StructField("bn", BinaryType, true) ::
            StructField("arr", ArrayType(IntegerType, true), true) ::
            StructField("map", MapType(StringType, StringType, true), true) ::
            StructField("struct", StructType(
              List(
                StructField("favorite_color", StringType, true),
                StructField("age", IntegerType, true)
              )
            ), true) :: Nil
        )

        import java.io.{ByteArrayOutputStream, ObjectOutputStream}
        val serialise = (value: Any) => {
          val stream: ByteArrayOutputStream = new ByteArrayOutputStream()
          val oos = new ObjectOutputStream(stream)
          oos.writeObject(value)
          oos.close()
          stream.toByteArray
        }

        val dataList = Seq(
          Row(
            true,
            7.toByte,
            15.toShort,
            3743,
            327828732L,
            5F,
            123.5,
            Decimal("1208484888.8474763788847476378884747637"),
            java.sql.Timestamp.valueOf("2020-06-29 22:41:30"),
            java.sql.Date.valueOf("2020-06-29"),
            "str",
            serialise("strBinary"),
            Array(1, 2, 3),
            Map("aguila" -> "Colombia", "modelo" -> "Mexico"),
            Row("blue", 45)),
          Row(
            false,
            8.toByte,
            9.toShort,
            3742,
            32728732L,
            10F,
            12.5,
            Decimal("1208484889.8474763788847476378884747637"),
            java.sql.Timestamp.valueOf("2020-06-30 22:41:30"),
            java.sql.Date.valueOf("2020-06-30"),
            "str2",
            serialise("strBinary2"),
            Array(1, 2, 3, 4),
            Map("aguila" -> "Colombia2", "modelo" -> "Mexico2"),
            Row("brown", 46)),
          Row(
            true,
            7.toByte,
            15.toShort,
            3743,
            327828732L,
            15F,
            123.5,
            Decimal("1208484888.8474763788847476378884747637"),
            java.sql.Timestamp.valueOf("2020-06-29 22:41:30"),
            java.sql.Date.valueOf("2020-06-29"),
            "str",
            serialise("strBinary"),
            Array(1, 2, 3),
            Map("aguila" -> "Colombia", "modelo" -> "Mexico"),
            Row("blue", 45)),
          Row(
            false,
            8.toByte,
            9.toShort,
            3742,
            32728732L,
            0F,
            12.5,
            Decimal("1208484889.8474763788847476378884747637"),
            java.sql.Timestamp.valueOf("2020-06-30 22:41:30"),
            java.sql.Date.valueOf("2020-06-30"),
            "str2",
            serialise("strBinary2"),
            Array(1, 2, 3, 4),
            Map("aguila" -> "Colombia2", "modelo" -> "Mexico2"),
            Row("brown", 46)),
          Row(
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null),
          Row(
            false,
            2.toByte,
            22.toShort,
            37342,
            327228732L,
            Float.NaN,
            Double.NegativeInfinity,
            Decimal("1228384889.8474763788847476378884747637"),
            java.sql.Timestamp.valueOf("2020-07-30 22:41:30"),
            java.sql.Date.valueOf("2020-07-30"),
            "str3",
            serialise("strBinary3"),
            Array(1, 2, 3, 4, 5),
            Map("aguila" -> "Colombia2", "modelo" -> "Mexico3"),
            Row("brown", 47)),
          Row(
            false,
            2.toByte,
            22.toShort,
            37342,
            327228732L,
            Float.PositiveInfinity,
            Double.NegativeInfinity,
            Decimal("1228384889.8474763788847476378884747637"),
            java.sql.Timestamp.valueOf("2020-07-30 22:41:30"),
            java.sql.Date.valueOf("2020-07-30"),
            "str3",
            serialise("strBinary3"),
            Array(1, 2, 3, 4, 5),
            Map("aguila" -> "Colombia2", "modelo" -> "Mexico3"),
            Row("brown", 47))
        )

        var df = sparkSession.createDataFrame(
          sparkSession.sparkContext.parallelize(dataList),
          schema
        )
        df = df.withColumn(target,
          functions.round(functions.rand(10) * lit(1)))
        df.show()
*/
/*
        val nVal = 5000
        val nTargetBins = 100
        var df = sparkSession.read.format("parquet")
          .load(f"test-data/features_int_1k_$nVal.parquet")
        df = df.withColumn(target,
          functions.round(functions.rand(10) * lit(nTargetBins-1)))
*/
        df.limit(nRowLimit)

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

        // convert numerics to doubles, while cleaning up positive and negative infinity values
        val castSelects = df.schema.map(c => { //TODO: fix for Date and Decimal Types
          if (c.dataType == FloatType) {
            when(col(c.name).isin(Float.PositiveInfinity, Float.NegativeInfinity),
              Float.NaN).otherwise(col(c.name)).alias(c.name)
          } else if (c.dataType == DoubleType) {
            when(col(c.name).isin(Double.PositiveInfinity, Double.NegativeInfinity),
              Double.NaN).otherwise(col(c.name)).alias(c.name)
          } else if (integralTypes.contains(c.dataType)){
            col(c.name).alias(c.name)
          } else if (numericTypes.contains(c.dataType)){
            col(c.name).cast(DoubleType).alias(c.name)
          } else {
            col(c.name).alias(c.name)
          }
        })
        df = df.select(castSelects: _*)
        df.persist(StorageLevel.MEMORY_AND_DISK_SER)


        // stats for feature selection and binning (a bit faster than deequ profiling (5-50%)
        // but still slow. Should be changed to rdd-based aggregations!!!
        val numStatsAggs = df.schema.filter(c => numericTypes.contains(c.dataType))
          .flatMap(c => {
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
        val otherStatsAggs = df.schema.filterNot(c => numericTypes.contains(c.dataType))
          .flatMap (c => Seq(
          approx_count_distinct(col(c.name)).alias(c.name+"_dist"),
          count(when(col(c.name).isNull, lit(1))).alias(c.name + "_count_null")))
        val generalCount = Seq(count(lit(1)).alias("_count"))
        val stats = df.select(numStatsAggs ++ otherStatsAggs ++
          generalCount: _*).first()

        //println(stats)

        // select features (simple low variance, low completeness, high distinctness filter)
        val selectedColumns = df.schema.filter(kv => kv.name != target).filterNot(c => {
          val distinct = stats.getAs[Long](c.name+"_dist")
          val count = stats.getAs[Long]("_count")
          val countNull = stats.getAs[Long](c.name+"_count_null")
          val countSafe = if (count == 0) 1 else count
          val noVariance = distinct == 1
          val lowCompleteness = (count - countNull) / countSafe.toDouble < completenessThreshold
          if (numericTypes.contains(c.dataType)) {
            val stddev = stats.getAs[Double](c.name+"_stddev")
            val mean = stats.getAs[Double](c.name+"_mean")
            val meanSafe = if (mean==0) 1e-100 else mean
            val lowVariance = (stddev/meanSafe)*(stddev/meanSafe) < normalizedVarianceThreshold
            if (integralTypes.contains(c.dataType)) {
              val highDistinctness = (distinct / countSafe.toDouble) > distinctnessThresholdIntegral
              noVariance || lowCompleteness || lowVariance || highDistinctness
            } else {
              noVariance || lowCompleteness || lowVariance
            }
          } else {
            val distinctness = (distinct / countSafe.toDouble)
            val highDistinctness = distinctness > distinctnessThresholdOther
            noVariance || lowCompleteness || highDistinctness
          }
        }).map(c => c.name) :+ target

        df = df.select(selectedColumns.map(c => col(c)).toArray: _*)

        val durationLoadingAndStats = (System.nanoTime - tLoadingAndStats) / 1e9d
        println(f"loading, stats, pre-filter x $durationLoadingAndStats")

        val tHashingBucketing = System.nanoTime

        // creating some lookup tables
        val selectedIndexes = ListMap(df.schema.fields
          .map { _.name }
          .zipWithIndex.toSeq: _*)
        val selectedTypes = df.schema
          .map { info => info.name -> info.dataType }
          .toMap
        println("selected features and indexes")
        println(selectedIndexes.map(i => (i._1, i._2 + 1)))
        println(selectedTypes.map(i => (i._1, i._2)))

        val bucketingLookupsBuf = ArrayBuffer[(String, (Double, Double, Long))]()
        selectedTypes.filter(kv => numericTypes.contains(kv._2)).foreach(c => {
          val min = stats.getAs[Double](c._1+"_min")
          val max = stats.getAs[Double](c._1+"_max")
          val dist = stats.getAs[Long](c._1+"_dist")
          if (dist > discretizationTreshold) {
            // add bucketing parameters: min, range, number of buckets
            bucketingLookupsBuf.append((c._1, (min, max-min, Math.min(nBuckets, dist))))
          }
        })
        val bucketingLookups = bucketingLookupsBuf.toMap
        println("bucketing lookups")
        println(bucketingLookups)

        val hashingColumns = selectedIndexes.filterNot(kv => {
          bucketingLookups.contains(kv._1) || byteCompatibleTypes.contains(selectedTypes(kv._1))
        })
        println("hashing columns")
        println(hashingColumns)

        // Columns which are either bucketizable or byte compatible are converted to bytes,
        // all others are hashed.
        val hashingBucketingAndByteCompatibleConversions = selectedIndexes.map(kv => {
           if (bucketingLookups.contains(kv._1)) {
             val b = bucketingLookups(kv._1)
             // put Null/NaN values in 0 bucket.. Put other in following n-1 buckets
             when(!col(kv._1).isNull && !col(kv._1).isNaN,
               ((col(kv._1).cast(DoubleType) - b._1) / b._2) * (b._3 - 1) + 1)
             .otherwise(lit(0)).cast(ByteType).alias(kv._1)
           } else if (byteCompatibleTypes.contains(selectedTypes(kv._1))) {
             col(kv._1).cast(ByteType).alias(kv._1)
           } else {
             xxhash64(col(kv._1)).alias(kv._1)
           }
        }).toSeq

        val hashedTableDf = df.select(hashingBucketingAndByteCompatibleConversions: _*)
        //hashedTableDf.show()
        val hashedTable = hashedTableDf.rdd
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

        println("hashing column lookups")
        //println(hashingColumnsLookups)

        val durationFrequentItemSketches = (System.nanoTime - tFrequentItemSketches) / 1e9d
        println(f"frequent item sketches x $durationFrequentItemSketches")

        val tColumnarBinary = System.nanoTime

        val nAllFeatures = selectedIndexes.size
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
                  values.getByte(dfIdx)
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

        val indexToFeatures = selectedIndexes.map(kv => kv._2 -> kv._1)

        MrmrSelector.trainColumnar(columnarData,  nSelectFeatures,  nAllFeatures, indexToFeatures)

        val durationSelection = (System.nanoTime - tSelection) / 1e9d
        println(f"fastmrmr x $durationSelection")
      }
  }
}
