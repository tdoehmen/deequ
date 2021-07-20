/**
 * Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import com.amazon.deequ.analyzers.{DataTypeInstances, KLLParameters}
import com.amazon.deequ.metrics.{BucketDistribution, BucketValue, Distribution, DistributionValue}
import com.amazon.deequ.profiles.{ColumnProfiler, NumericColumnProfile, StandardColumnProfile}
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

    "return correct NumericColumnProfiles for numeric columns with correct DataType" in
      withSparkSession { session =>

        val data = getDfWithNumericFractionalValues(session)

        val actualColumnProfile = ColumnProfiler.profile_approx(data, Option(Seq("att1")))
          .profiles("att1")

        val expectedColumnProfile = NumericColumnProfile(
          "att1",
          1.0,
          None,
          None,
          None,
          6,
          DataTypeInstances.Fractional,
          false,
          Map.empty,
          None,
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

    "return correct NumericColumnProfiles with uniqueness, distinctness and entropy " in
      withSparkSession { session =>

        val data = getDfWithNumericFractionalValues(session)

        val actualColumnProfile = ColumnProfiler.profile_approx(data, Option(Seq("att1")),
          uniquenessCols = Seq("att1")).profiles("att1")

        val expectedColumnProfile = NumericColumnProfile(
          "att1",
          1.0,
          Some(1.0),
          Some(1.791759469228055),
          Some(1.0),
          6,
          DataTypeInstances.Fractional,
          false,
          Map.empty,
          None,
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

        val actualColumnProfile = ColumnProfiler.profile_approx(data, Option(Seq("item")),
          uniquenessCols = Seq("item"), histogram = true).profiles("item")

        val expectedColumnProfile = StandardColumnProfile(
          "item",
          1.0,
          Some(1.0),
          Some(1.791759469228055),
          Some(1.0),
          6,
          DataTypeInstances.String,
          false,
          Map.empty,
          Some(Distribution(Map("4" -> DistributionValue(1, 0.16666666666666666),
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

