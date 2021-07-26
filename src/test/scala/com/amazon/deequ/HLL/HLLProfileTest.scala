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

package com.amazon.deequ.HLL

import com.amazon.deequ.SparkContextSpec
import com.amazon.deequ.analyzers.{ApproxCountDistinct, InMemoryStateProvider, Size}
import com.amazon.deequ.profiles.{ColumnProfiler, ColumnProfiles}
import com.amazon.deequ.utils.FixtureSupport
import org.scalatest.{Matchers, WordSpec}
import org.apache.spark.sql.catalyst.expressions.aggregate.DeequHyperLogLogPlusPlusUtils
import org.apache.spark.sql.functions.{lit, udf}
import org.apache.spark.sql.types.LongType

class HLLProfileTest extends WordSpec with Matchers with SparkContextSpec
  with FixtureSupport {

  "Column Profiler" should {

    "return correct HLL properties" in
      withSparkSession { session =>

        val nRows = 100
        val nVal = 100

        import session.implicits._
        import org.apache.spark.sql.functions

        var data = session.sparkContext.range(0,nRows).toDF().select(functions.col("value"))
        data = data.withColumnRenamed("value","att1")
        data = data.withColumn("att2",functions.round(functions.rand(0)*lit(nVal)).cast(LongType))

        println(data.schema)
        data.show(10)
        println("did not contain: "+data.filter("att2 >= 1000000").count())

        val stateRepository = InMemoryStateProvider()
        val actualColumnProfile = ColumnProfiler.profileOptimized(data, Option(Seq("att1","att2")), stateRepository = stateRepository)

        val profile1 = actualColumnProfile.profiles.get("att1")
        val profile2 = actualColumnProfile.profiles.get("att2")

        println("profile1.get.approximateNumDistinctValues: "+profile1.get.approximateNumDistinctValues)
        println("profile2.get.approximateNumDistinctValues: "+profile2.get.approximateNumDistinctValues)

        val state1 = stateRepository.load(ApproxCountDistinct("att1"))
        val state2 = stateRepository.load(ApproxCountDistinct("att2"))

        val words1 = state1.get.words
        val words2 = state2.get.words

        println("words1: "+words1.mkString(", "))
        println("words2: "+words2.mkString(", "))

        val count1 = DeequHyperLogLogPlusPlusUtils.count(words1)
        val count2 = DeequHyperLogLogPlusPlusUtils.count(words2)

        println("count1: "+count1)
        println("count2: "+count2)
        println("Size() is defined: "+stateRepository.load(Size()).isDefined)

        val wordsMerged = DeequHyperLogLogPlusPlusUtils.merge(words1,words2)
        val countMerged = DeequHyperLogLogPlusPlusUtils.count(wordsMerged)

        println("countMerged: "+countMerged)

        val profiles = actualColumnProfile.profiles.map{pro => pro._2}.toSeq
        val json_profile = ColumnProfiles.toJson(profiles)
        println(json_profile)
      }

  }
}


