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

package com.amazon.deequ.profiles

import com.amazon.deequ.SparkContextSpec
import com.amazon.deequ.utils.FixtureSupport
import org.scalatest.{Matchers, color}
import org.scalatest.wordspec.AnyWordSpec
class ColumnProfilerNaNTest extends AnyWordSpec with Matchers with SparkContextSpec with FixtureSupport {

  "Column Profiler NaN Test" should {
    "return results for data frame with NaN and null values without failure" in withSparkSession {
      sparkSession =>
        val df = getDfWithNas(sparkSession)

        val runner: ColumnProfilerRunBuilder = new ColumnProfilerRunner()
          .onData(df)
          .withCorrelation(true, 50)
          .withHistogram(true, 20)
          .withExactUniqueness (true)

        val result = runner.run()

        val matches = result.profiles.map { case (colname: String, profile: ColumnProfile) =>
          val nacount = df.filter(df(colname).isNull || df(colname).isNaN).count()
          val nacount_profile = result.numRecords - scala.math.round(profile.completeness *
                                                                     result.numRecords)
          nacount == nacount_profile
        }.toSeq

        assert(matches.forall(_ == true))
    }
  }
}