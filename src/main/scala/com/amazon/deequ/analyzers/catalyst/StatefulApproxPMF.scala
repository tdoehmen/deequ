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

package org.apache.spark.sql

import com.amazon.deequ.analyzers.{KLLSketch, KLLState, QuantileNonSample}
import com.amazon.deequ.metrics.BucketValue
import org.apache.spark.sql.expressions.MutableAggregationBuffer
import org.apache.spark.sql.types.{ArrayType, BinaryType, DataType, DoubleType, LongType, StructField, StructType}

import scala.collection.mutable.{ArrayBuffer, ListBuffer}

protected[sql] class StatefulApproxPMF(numberOfBuckets: Int, start: Option[Double],
                                       end: Option[Double])
  extends StatefulKLLSketch(KLLSketch.DEFAULT_SKETCH_SIZE, KLLSketch.DEFAULT_SHRINKING_FACTOR){

  val COUNT_POS = 3

  override def bufferSchema: StructType = StructType(StructField("data", BinaryType) ::
    StructField("minimum", DoubleType) :: StructField("maximum", DoubleType) :: StructField
    ("count", LongType) :: Nil)

  override def dataType: DataType = StructType(StructField("dist", ArrayType(DoubleType)) ::
    StructField("min", DoubleType) :: StructField("max", DoubleType) :: StructField
  ("count", LongType) :: Nil)

  override def initialize(buffer: MutableAggregationBuffer): Unit = {
    super.initialize(buffer)

    buffer(COUNT_POS) = 0L
  }

  override def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
    if (input.isNullAt(OBJECT_POS)) {
      return
    }
    super.update(buffer, input)

    buffer(COUNT_POS) = buffer.getLong(COUNT_POS) + 1L
  }

  override def merge(buffer1: MutableAggregationBuffer, buffer2: Row): Unit = {
    if (buffer2.isNullAt(OBJECT_POS)) {
      return
    }

    super.merge(buffer1, buffer2)

    buffer1(COUNT_POS) = buffer1.getLong(COUNT_POS) + buffer2.getLong(COUNT_POS)
  }

  override def evaluate(buffer: Row): Any = {
    val finalSketch = deserialize(buffer.getAs[Array[Byte]](OBJECT_POS))
    val start = this.start.getOrElse(buffer.getDouble(MIN_POS))
    val end = this.end.getOrElse(buffer.getDouble(MAX_POS))

    val count = buffer.getLong(COUNT_POS)

    var bucketsList = new ListBuffer[Double]()
    for (i <- 0 until numberOfBuckets) {
      val lowBound = start + (end - start) * i / numberOfBuckets.toDouble
      val highBound = start + (end - start) * (i + 1) / numberOfBuckets.toDouble
      if (i == numberOfBuckets - 1) {
        bucketsList += (finalSketch.getRank(highBound) -
                        finalSketch.getRankExclusive(lowBound)) / count.toDouble
      } else {
        bucketsList += (finalSketch.getRankExclusive(highBound) -
                        finalSketch.getRankExclusive(lowBound)) / count.toDouble
      }
    }

    (bucketsList, buffer.getDouble(MIN_POS), buffer.getDouble(MAX_POS), count)

  }

}
