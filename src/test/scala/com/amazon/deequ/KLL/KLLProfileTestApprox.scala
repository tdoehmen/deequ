/**
 * Copyright 2021 Logical Clocks AB. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not
 * use this file except in compliance with the License. A copy of the License
 * is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on
 * an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 * express or implied. See the License for the specific language governing
 * permissions and limitations under the License.
 *
 */

package com.amazon.deequ.KLL

import com.amazon.deequ.SparkContextSpec
import com.amazon.deequ.analyzers.{DataTypeInstances, KLLParameters, KLLSketch}
import com.amazon.deequ.metrics.{BucketDistribution, BucketValue, Distribution, DistributionValue}
import com.amazon.deequ.profiles.{ColumnProfiler, ColumnProfiles, NumericColumnProfile, StandardColumnProfile}
import com.amazon.deequ.utils.FixtureSupport
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.scalatest.{Matchers, WordSpec}

class KLLProfileTestApprox extends WordSpec with Matchers with SparkContextSpec
  with FixtureSupport {

  def assertProfilesEqual(expected: NumericColumnProfile, actual: NumericColumnProfile): Unit = {

    assert(expected.column == actual.column)
    assert(expected.completeness == actual.completeness)
    assert(math.abs(expected.approximateNumDistinctValues -
      actual.approximateNumDistinctValues) <= 1)
    assert(expected.uniqueness == actual.uniqueness)
    assert(expected.distinctness == actual.distinctness)
    assert(expected.entropy == actual.entropy)
    assert(expected.dataType == actual.dataType)
    assert(expected.isDataTypeInferred == expected.isDataTypeInferred)
    assert(expected.typeCounts == actual.typeCounts)
    assert(expected.histogram == actual.histogram)
    assert(expected.mean == actual.mean)
    assert(expected.maximum == actual.maximum)
    assert(expected.minimum == actual.minimum)
    assert(expected.sum == actual.sum)
    assert(expected.stdDev == actual.stdDev)
    assert(expected.kll == actual.kll)
    assert(expected.approxPercentiles == actual.approxPercentiles)
    assert(expected.correlation == actual.correlation)
  }


  def assertStandardProfilesEqual(expected: StandardColumnProfile,
                                  actual: StandardColumnProfile): Unit = {

    assert(expected.column == actual.column)
    assert(expected.completeness == actual.completeness)
    assert(expected.uniqueness == actual.uniqueness)
    assert(expected.distinctness == actual.distinctness)
    assert(expected.entropy == actual.entropy)
    assert(math.abs(expected.approximateNumDistinctValues -
      actual.approximateNumDistinctValues) <= 1)
    assert(expected.dataType == actual.dataType)
    assert(expected.isDataTypeInferred == expected.isDataTypeInferred)
    assert(expected.typeCounts == actual.typeCounts)
    assert(expected.histogram == actual.histogram)
  }


  "Column Profiler" should {

    "return correct NumericColumnProfiles for decimal column" in
      withSparkSession { session =>

        val data = getDfWithDecimalFractionalValues(session)

        val actualColumnProfile = ColumnProfiler.profileOptimized(data, Option(Seq("att1",
          "att2")), kllParameters = Some(KLLParameters(KLLSketch.DEFAULT_SKETCH_SIZE, KLLSketch
          .DEFAULT_SHRINKING_FACTOR, 20)), histogram = true)
          .profiles("att1")

        val expectedColumnProfile = NumericColumnProfile(
          "att1",
          1.0,
          None,
          None,
          None,
          6,
          None,
          DataTypeInstances.Decimal,
          false,
          Map.empty,
          Some(Distribution(Map[String, DistributionValue](
            "4.000000000000000000" -> DistributionValue(1, 0.16666666666666666),
            "1.000000000000000000" -> DistributionValue(1, 0.16666666666666666),
            "5.000000000000000000" -> DistributionValue(1, 0.16666666666666666),
            "6.000000000000000000" -> DistributionValue(1, 0.16666666666666666),
            "2.000000000000000000" -> DistributionValue(1, 0.16666666666666666),
            "3.000000000000000000" -> DistributionValue(1, 0.16666666666666666)), 6)),
          Some(BucketDistribution(List(BucketValue(1.0, 1.25, 1),
            BucketValue(1.25, 1.5, 0),
            BucketValue(1.5, 1.75, 0),
            BucketValue(1.75, 2.0, 0),
            BucketValue(2.0, 2.25, 1),
            BucketValue(2.25, 2.5, 0),
            BucketValue(2.5, 2.75, 0),
            BucketValue(2.75, 3.0, 0),
            BucketValue(3.0, 3.25, 1),
            BucketValue(3.25, 3.5, 0),
            BucketValue(3.5, 3.75, 0),
            BucketValue(3.75, 4.0, 0),
            BucketValue(4.0, 4.25, 1),
            BucketValue(4.25, 4.5, 0),
            BucketValue(4.5, 4.75, 0),
            BucketValue(4.75, 5.0, 0),
            BucketValue(5.0, 5.25, 1),
            BucketValue(5.25, 5.5, 0),
            BucketValue(5.5, 5.75, 0),
            BucketValue(5.75, 6.0, 1)),
            List(0.64, 2048.0),
            Array(Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)))),
          Some(3.5),
          Some(6.0),
          Some(1.0),
          Some(21.0),
          Some(1.707825127659933),
          Some(Seq(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0,
            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0,
            4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
            4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0,
            5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
            5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0,
            6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0)),
          Some(Map[String, Double]("att1" -> 1.0, "att2" -> 0.9263710192499128))
        )

        assertProfilesEqual(expectedColumnProfile,
          actualColumnProfile.asInstanceOf[NumericColumnProfile])
      }

    "return correct NumericColumnProfiles for numeric columns with correct DataType" in
      withSparkSession { session =>

        val data = getDfWithNumericFractionalValues(session)

        val actualColumnProfile = ColumnProfiler.profileOptimized(data, Option(Seq("att1",
          "att2")), kllParameters = Some(KLLParameters(KLLSketch.DEFAULT_SKETCH_SIZE, KLLSketch
          .DEFAULT_SHRINKING_FACTOR, 20)), histogram = true)
          .profiles("att1")

        val expectedColumnProfile = NumericColumnProfile(
          "att1",
          1.0,
          None,
          None,
          None,
          6,
          None,
          DataTypeInstances.Fractional,
          false,
          Map.empty,
          actualColumnProfile.histogram,
          Some(BucketDistribution(List(BucketValue(1.0, 1.25, 1),
            BucketValue(1.25, 1.5, 0),
            BucketValue(1.5, 1.75, 0),
            BucketValue(1.75, 2.0, 0),
            BucketValue(2.0, 2.25, 1),
            BucketValue(2.25, 2.5, 0),
            BucketValue(2.5, 2.75, 0),
            BucketValue(2.75, 3.0, 0),
            BucketValue(3.0, 3.25, 1),
            BucketValue(3.25, 3.5, 0),
            BucketValue(3.5, 3.75, 0),
            BucketValue(3.75, 4.0, 0),
            BucketValue(4.0, 4.25, 1),
            BucketValue(4.25, 4.5, 0),
            BucketValue(4.5, 4.75, 0),
            BucketValue(4.75, 5.0, 0),
            BucketValue(5.0, 5.25, 1),
            BucketValue(5.25, 5.5, 0),
            BucketValue(5.5, 5.75, 0),
            BucketValue(5.75, 6.0, 1)),
            List(0.64, 2048.0),
            Array(Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)))),
          Some(3.5),
          Some(6.0),
          Some(1.0),
          Some(21.0),
          Some(1.707825127659933),
          Some(Seq(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0,
            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0,
            4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
            4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0,
            5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
            5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0,
            6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0)),
          Some(Map[String, Double]("att1" -> 1.0, "att2" -> 0.9263710192499128))
        )

        assertProfilesEqual(expectedColumnProfile,
          actualColumnProfile.asInstanceOf[NumericColumnProfile])
      }

    "return correct JSON for NumericColumnProfiles" in
      withSparkSession { session =>

        val data = getDfWithNumericFractionalValues(session)

        val profile = ColumnProfiler.profileOptimized(data, Option(Seq("att1", "att2")),
          kllParameters = Some(KLLParameters(KLLSketch.DEFAULT_SKETCH_SIZE, KLLSketch
          .DEFAULT_SHRINKING_FACTOR, 20)), histogram = true, exactUniqueness = true)
        val json_profile = ColumnProfiles.toJson(profile)
        val correct_profile = "{\"columns\":[{\"column\":\"att1\",\"dataType\":\"Fractional\"," +
          "\"isDataTypeInferred\":\"false\",\"completeness\":1.0,\"numRecordsNonNull\":6," +
          "\"numRecordsNull\":0,\"distinctness\":1.0,\"entropy\":1.791759469228055," +
          "\"uniqueness\":1.0,\"approximateNumDistinctValues\":6,\"exactNumDistinctValues\":6," +
          "\"histogram\":[{\"value\":\"1.0\",\"count\":1,\"ratio\":0.16666666666666666}," +
          "{\"value\":\"2.0\",\"count\":1,\"ratio\":0.16666666666666666},{\"value\":\"3.0\"," +
          "\"count\":1,\"ratio\":0.16666666666666666},{\"value\":\"4.0\",\"count\":1," +
          "\"ratio\":0.16666666666666666},{\"value\":\"5.0\",\"count\":1," +
          "\"ratio\":0.16666666666666666},{\"value\":\"6.0\",\"count\":1," +
          "\"ratio\":0.16666666666666666}],\"mean\":3.5,\"maximum\":6.0,\"minimum\":1.0," +
          "\"sum\":21.0,\"stdDev\":1.707825127659933,\"correlations\":[{\"column\":\"att2\"," +
          "\"correlation\":0.9263710192499128},{\"column\":\"att1\",\"correlation\":1.0}]," +
          "\"kll\":{\"buckets\":[{\"low_value\":1.0,\"high_value\":1.25,\"count\":1," +
          "\"ratio\":0.16666666666666666},{\"low_value\":1.25,\"high_value\":1.5,\"count\":0," +
          "\"ratio\":0.0},{\"low_value\":1.5,\"high_value\":1.75,\"count\":0,\"ratio\":0.0}," +
          "{\"low_value\":1.75,\"high_value\":2.0,\"count\":0,\"ratio\":0.0},{\"low_value\":2.0," +
          "\"high_value\":2.25,\"count\":1,\"ratio\":0.16666666666666666},{\"low_value\":2.25," +
          "\"high_value\":2.5,\"count\":0,\"ratio\":0.0},{\"low_value\":2.5,\"high_value\":2.75," +
          "\"count\":0,\"ratio\":0.0},{\"low_value\":2.75,\"high_value\":3.0,\"count\":0," +
          "\"ratio\":0.0},{\"low_value\":3.0,\"high_value\":3.25,\"count\":1," +
          "\"ratio\":0.16666666666666666},{\"low_value\":3.25,\"high_value\":3.5,\"count\":0," +
          "\"ratio\":0.0},{\"low_value\":3.5,\"high_value\":3.75,\"count\":0,\"ratio\":0.0}," +
          "{\"low_value\":3.75,\"high_value\":4.0,\"count\":0,\"ratio\":0.0},{\"low_value\":4.0," +
          "\"high_value\":4.25,\"count\":1,\"ratio\":0.16666666666666666},{\"low_value\":4.25," +
          "\"high_value\":4.5,\"count\":0,\"ratio\":0.0},{\"low_value\":4.5,\"high_value\":4.75," +
          "\"count\":0,\"ratio\":0.0},{\"low_value\":4.75,\"high_value\":5.0,\"count\":0," +
          "\"ratio\":0.0},{\"low_value\":5.0,\"high_value\":5.25,\"count\":1," +
          "\"ratio\":0.16666666666666666},{\"low_value\":5.25,\"high_value\":5.5,\"count\":0," +
          "\"ratio\":0.0},{\"low_value\":5.5,\"high_value\":5.75,\"count\":0,\"ratio\":0.0}," +
          "{\"low_value\":5.75,\"high_value\":6.0,\"count\":1,\"ratio\":0.16666666666666666}]," +
          "\"sketch\":{\"parameters\":{\"c\":0.64,\"k\":2048.0},\"data\":\"[[1.0,2.0,3.0,4.0,5.0," +
          "6.0]]\"}},\"approxPercentiles\":[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0," +
          "1.0,1.0,1.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,3.0," +
          "3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,4.0,4.0,4.0,4.0,4.0,4.0," +
          "4.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0," +
          "5.0,5.0,5.0,5.0,5.0,5.0,5.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0," +
          "6.0,6.0]},{\"column\":\"att2\",\"dataType\":\"Fractional\"," +
          "\"isDataTypeInferred\":\"false\",\"completeness\":1.0,\"numRecordsNonNull\":6," +
          "\"numRecordsNull\":0,\"distinctness\":0.6666666666666666,\"entropy\":1.242453324894," +
          "\"uniqueness\":0.5,\"approximateNumDistinctValues\":4,\"exactNumDistinctValues\":4," +
          "\"histogram\":[{\"value\":\"0.0\",\"count\":3,\"ratio\":0.5},{\"value\":\"5.0\"," +
          "\"count\":1,\"ratio\":0.16666666666666666},{\"value\":\"6.0\",\"count\":1," +
          "\"ratio\":0.16666666666666666},{\"value\":\"7.0\",\"count\":1," +
          "\"ratio\":0.16666666666666666}],\"mean\":3.0,\"maximum\":7.0,\"minimum\":0.0," +
          "\"sum\":18.0,\"stdDev\":3.0550504633038935,\"correlations\":[{\"column\":\"att2\"," +
          "\"correlation\":1.0},{\"column\":\"att1\",\"correlation\":0.9263710192499128}]," +
          "\"kll\":{\"buckets\":[{\"low_value\":0.0,\"high_value\":0.35,\"count\":3," +
          "\"ratio\":0.5},{\"low_value\":0.35,\"high_value\":0.7,\"count\":0,\"ratio\":0.0}," +
          "{\"low_value\":0.7,\"high_value\":1.05,\"count\":0,\"ratio\":0.0},{\"low_value\":1.05," +
          "\"high_value\":1.4,\"count\":0,\"ratio\":0.0},{\"low_value\":1.4,\"high_value\":1.75," +
          "\"count\":0,\"ratio\":0.0},{\"low_value\":1.75,\"high_value\":2.1,\"count\":0," +
          "\"ratio\":0.0},{\"low_value\":2.1,\"high_value\":2.45,\"count\":0,\"ratio\":0.0}," +
          "{\"low_value\":2.45,\"high_value\":2.8,\"count\":0,\"ratio\":0.0},{\"low_value\":2.8," +
          "\"high_value\":3.15,\"count\":0,\"ratio\":0.0},{\"low_value\":3.15,\"high_value\":3.5," +
          "\"count\":0,\"ratio\":0.0},{\"low_value\":3.5,\"high_value\":3.85,\"count\":0," +
          "\"ratio\":0.0},{\"low_value\":3.85,\"high_value\":4.2,\"count\":0,\"ratio\":0.0}," +
          "{\"low_value\":4.2,\"high_value\":4.55,\"count\":0,\"ratio\":0.0},{\"low_value\":4.55," +
          "\"high_value\":4.9,\"count\":0,\"ratio\":0.0},{\"low_value\":4.9,\"high_value\":5.25," +
          "\"count\":1,\"ratio\":0.16666666666666666},{\"low_value\":5.25,\"high_value\":5.6," +
          "\"count\":0,\"ratio\":0.0},{\"low_value\":5.6,\"high_value\":5.95,\"count\":0," +
          "\"ratio\":0.0},{\"low_value\":5.95,\"high_value\":6.3,\"count\":1," +
          "\"ratio\":0.16666666666666666},{\"low_value\":6.3,\"high_value\":6.65,\"count\":0," +
          "\"ratio\":0.0},{\"low_value\":6.65,\"high_value\":7.0,\"count\":1," +
          "\"ratio\":0.16666666666666666}],\"sketch\":{\"parameters\":{\"c\":0.64,\"k\":2048.0}," +
          "\"data\":\"[[0.0,0.0,0.0,5.0,6.0,7.0]]\"}},\"approxPercentiles\":[0.0,0.0,0.0,0.0,0.0," +
          "0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0," +
          "0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0," +
          "0.0,0.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,6.0,6.0," +
          "6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,7.0,7.0,7.0,7.0,7.0,7.0," +
          "7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0]}]}"
        assert(json_profile == correct_profile)
      }

    "return correct NumericColumnProfiles with uniqueness, distinctness and entropy " in
      withSparkSession { session =>

        val data = getDfWithNumericFractionalValues(session)

        val actualColumnProfile = ColumnProfiler.profileOptimized(data, Option(Seq("att1")),
          exactUniqueness = true, exactUniquenessCols = Some(Seq("att1")), kllParameters = Some
          (KLLParameters(KLLSketch.DEFAULT_SKETCH_SIZE, KLLSketch
            .DEFAULT_SHRINKING_FACTOR, 20)), histogram = true).profiles("att1")

        val expectedColumnProfile = NumericColumnProfile(
          "att1",
          1.0,
          Some(1.0),
          Some(1.791759469228055),
          Some(1.0),
          6,
          Some(6),
          DataTypeInstances.Fractional,
          false,
          Map.empty,
          actualColumnProfile.histogram,
          Some(BucketDistribution(List(BucketValue(1.0, 1.25, 1),
            BucketValue(1.25, 1.5, 0),
            BucketValue(1.5, 1.75, 0),
            BucketValue(1.75, 2.0, 0),
            BucketValue(2.0, 2.25, 1),
            BucketValue(2.25, 2.5, 0),
            BucketValue(2.5, 2.75, 0),
            BucketValue(2.75, 3.0, 0),
            BucketValue(3.0, 3.25, 1),
            BucketValue(3.25, 3.5, 0),
            BucketValue(3.5, 3.75, 0),
            BucketValue(3.75, 4.0, 0),
            BucketValue(4.0, 4.25, 1),
            BucketValue(4.25, 4.5, 0),
            BucketValue(4.5, 4.75, 0),
            BucketValue(4.75, 5.0, 0),
            BucketValue(5.0, 5.25, 1),
            BucketValue(5.25, 5.5, 0),
            BucketValue(5.5, 5.75, 0),
            BucketValue(5.75, 6.0, 1)),
            List(0.64, 2048.0),
            Array(Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)))),
          Some(3.5),
          Some(6.0),
          Some(1.0),
          Some(21.0),
          Some(1.707825127659933),
          Some(Seq(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0,
            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0,
            4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
            4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0,
            5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
            5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0,
            6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0)),
          Some(Map[String, Double]("att1" -> 1.0))
        )

        assertProfilesEqual(expectedColumnProfile,
          actualColumnProfile.asInstanceOf[NumericColumnProfile])
      }

    "return correct StandardColumnProfile plus histogram for String column" in
      withSparkSession { session =>

        val data = getDfWithNumericFractionalValues(session)

        val actualColumnProfile = ColumnProfiler.profileOptimized(data, Option(Seq("item")),
          exactUniqueness = true, exactUniquenessCols = Some(Seq("item")), histogram = true)
          .profiles("item")

        val expectedColumnProfile = StandardColumnProfile(
          "item",
          1.0,
          Some(1.0),
          Some(1.791759469228055),
          Some(1.0),
          6,
          Some(6),
          DataTypeInstances.String,
          false,
          Map.empty,
          Some(Distribution(Map[String, DistributionValue](
            "4" -> DistributionValue(1, 0.16666666666666666),
            "5" -> DistributionValue(1, 0.16666666666666666),
            "6" -> DistributionValue(1, 0.16666666666666666),
            "1" -> DistributionValue(1, 0.16666666666666666),
            "2" -> DistributionValue(1, 0.16666666666666666),
            "3" -> DistributionValue(1, 0.16666666666666666)), 6))
        )

        assertStandardProfilesEqual(expectedColumnProfile,
          actualColumnProfile.asInstanceOf[StandardColumnProfile])
      }

    "return correct NumericColumnProfiles With KLL for numeric columns with correct DataType" in
      withSparkSession { session =>

        val data = getDfWithNumericFractionalValuesForKLL(session)

        val actualColumnProfile = ColumnProfiler.profile(data, Option(Seq("att1")), false, 1,
          kllProfiling = true,
          kllParameters = Option(KLLParameters(2, 0.64, 2)))
          .profiles("att1")

        val expectedColumnProfile = NumericColumnProfile(
          "att1",
          1.0,
          Some(1.0),
          Some(3.4011973816621546),
          Some(1.0),
          30,
          Some(30),
          DataTypeInstances.Fractional,
          false,
          Map.empty,
          None,
          Some(BucketDistribution(List(BucketValue(1.0, 15.5, 16),
            BucketValue(15.5, 30.0, 14)),
            List(0.64, 2.0),
            Array(Array(27.0, 28.0, 29.0, 30.0),
              Array(25.0),
              Array(1.0, 6.0, 10.0, 15.0, 19.0, 23.0)))),
          Some(15.5),
          Some(30.0),
          Some(1.0),
          Some(465.0),
          Some(8.65544144839919),
          Some(Seq(1.0, 1.0, 1.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0,
            6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 10.0, 10.0, 10.0,
            10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
            10.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0,
            15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 19.0, 19.0, 19.0,
            19.0, 19.0, 19.0, 19.0, 19.0, 19.0, 19.0, 19.0, 19.0,
            19.0, 23.0, 23.0, 23.0, 23.0, 23.0, 23.0, 23.0, 23.0,
            23.0, 23.0, 23.0, 23.0, 23.0, 25.0, 25.0, 25.0, 25.0,
            25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0,
            25.0, 27.0, 27.0, 27.0, 27.0, 27.0, 27.0, 28.0, 28.0,
            28.0, 28.0, 29.0, 29.0, 29.0, 30.0, 30.0, 30.0)),
          Some(Map[String, Double]("att1" -> 1.0)))

        assertProfilesEqual(expectedColumnProfile,
          actualColumnProfile.asInstanceOf[NumericColumnProfile])
      }

    "return KLL Sketches for ShortType columns" in withSparkSession { session =>
      val attribute = "attribute"
      val data = com.amazon.deequ.dataFrameWithColumn(
        attribute,
        ShortType,
        session,
        Row(1: Short),
        Row(2: Short),
        Row(3: Short),
        Row(4: Short),
        Row(5: Short),
        Row(6: Short),
        Row(null)
      )

      val actualColumnProfile = ColumnProfiler.profile(data,
        kllProfiling = true,
        kllParameters = Option(KLLParameters(2, 0.64, 2)))
        .profiles(attribute)
      val numericalProfile = actualColumnProfile.asInstanceOf[NumericColumnProfile]
      assert(numericalProfile.kll.isDefined)
      val kll = numericalProfile.kll
      assert(kll.get.buckets == List(BucketValue(1.0, 3.5, 4), BucketValue(3.5, 6.0, 2)))
      assert(kll.get.parameters == List(0.64, 2.0))
      assert(kll.get.data.length == 2)
      val target = Array(Array(5.0, 6.0), Array(1.0, 3.0))
      for (i <- kll.get.data.indices) {
        assert(kll.get.data(i).sameElements(target(i)))
      }
    }
  }
}

