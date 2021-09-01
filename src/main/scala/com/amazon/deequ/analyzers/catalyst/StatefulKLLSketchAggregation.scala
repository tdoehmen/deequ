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

package com.amazon.deequ.analyzers.catalyst

import com.amazon.deequ.analyzers.QuantileNonSample
import org.apache.spark.sql.expressions.Aggregator
import org.apache.spark.sql.{Encoder, Encoders, KLLStateInt}

import scala.collection.mutable.ArrayBuffer

class StatefulKLLSketchAggregation(sketchSize: Int, shrinkingFactor: Double) extends
  Aggregator[Array[Byte], KLLStateInt, Array[Byte]] {
  // A zero value for this aggregation. Should satisfy the property that any b + zero = b
  def zero: KLLStateInt = KLLStateInt(serialize(new QuantileNonSample[Double](sketchSize,
    shrinkingFactor)) , Int.MaxValue.toDouble, Int.MinValue.toDouble, new ArrayBuffer[Double], 0)

  // Combine two values to produce a new value. For performance, the function may modify `buffer`
  // and return it instead of constructing a new object
  def reduce(buffer: KLLStateInt, data: Array[Byte]): KLLStateInt = {
    merge(buffer, KLLStateInt.fromBytes(data))
  }

  // Merge two intermediate values
  def merge(b1: KLLStateInt, b2: KLLStateInt): KLLStateInt = {
    val leftSketch = deserialize(b1.sketch)
    val rightSketch = deserialize(b2.sketch)

    leftSketch.merge(rightSketch)
    b1.min = Math.min(b1.min, b2.min)
    b1.max = Math.max(b1.max, b2.max)
    b1.count = b1.count + b2.count
    b1.sketch = serialize(leftSketch)

    b1
  }

  // Transform the output of the reduction
  def finish(reduction: KLLStateInt): Array[Byte] = KLLStateInt.toBytes(reduction)
  // Specifies the Encoder for the intermediate value type
  def bufferEncoder: Encoder[KLLStateInt] = Encoders.product
  // Specifies the Encoder for the final output value type
  def outputEncoder: Encoder[Array[Byte]] = Encoders.BINARY

  def serialize(obj: QuantileNonSample[Double]): Array[Byte] = {
    KLLSketchSerializer.serializer.serialize(obj)
  }

  def deserialize(bytes: Array[Byte]): QuantileNonSample[Double] = {
    KLLSketchSerializer.serializer.deserialize(bytes)
  }
}

