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

import java.nio.ByteBuffer

import com.amazon.deequ.analyzers.QuantileNonSample
import com.amazon.deequ.analyzers.catalyst.KLLSketchSerializer
import com.google.common.primitives.{Doubles, Longs}
import org.apache.spark.sql.expressions.Aggregator

import scala.collection.mutable.ArrayBuffer

case class KLLStateInt(var sketch: Array[Byte], var min: Double, var
max: Double, var valueBuffer: ArrayBuffer[Double])

object KLLStateInt{
  def toBytes(kll: KLLStateInt): Array[Byte] = {
    val buffer2 = ByteBuffer.wrap(new Array(Doubles.BYTES + Doubles.BYTES + kll.sketch
      .length))
    buffer2.putDouble(kll.min)
    buffer2.putDouble(kll.max)
    buffer2.put(kll.sketch)
    buffer2.array()
  }

  def fromBytes(bytes: Array[Byte]): KLLStateInt = {
    val buffer = ByteBuffer.wrap(bytes)
    val min = buffer.getDouble
    val max = buffer.getDouble
    val kllBuffer = new Array[Byte](buffer.remaining())
    buffer.get(kllBuffer)
    KLLStateInt(kllBuffer, min, max, new ArrayBuffer())
  }
}

class StatefulKLLSketch2(sketchSize: Int, shrinkingFactor: Double, bufferSize: Int) extends
  Aggregator[Double, KLLStateInt, Array[Byte]] {
  // A zero value for this aggregation. Should satisfy the property that any b + zero = b
  def zero: KLLStateInt = KLLStateInt(serialize(new QuantileNonSample[Double](sketchSize, shrinkingFactor)), Int
    .MaxValue.toDouble, Int.MinValue.toDouble, new ArrayBuffer[Double])

  // Combine two values to produce a new value. For performance, the function may modify `buffer`
  // and return it instead of constructing a new object
  def reduce(buffer: KLLStateInt, data: Double): KLLStateInt = {
    if (data.isNaN) {
      return buffer
    }

    buffer.valueBuffer.append(data)

    if (buffer.valueBuffer.length > bufferSize) {
      val sketch = flushBuffer(buffer)
      buffer.sketch = serialize(sketch)
    }

    buffer
  }

  def flushBuffer(buffer: KLLStateInt): QuantileNonSample[Double] = {
    val sketch = deserialize(buffer.sketch)

    buffer.valueBuffer.foreach(value => {
      sketch.update(value)
      buffer.min = Math.min(buffer.min, value)
      buffer.max = Math.max(buffer.max, value)
    })

    buffer.valueBuffer = new ArrayBuffer[Double]
    sketch
  }

  // Merge two intermediate values
  def merge(b1: KLLStateInt, b2: KLLStateInt): KLLStateInt = {
    val leftSketch = flushBuffer(b1)
    val rightSketch = flushBuffer(b2)

    leftSketch.merge(rightSketch)
    b1.min = Math.min(b1.min, b2.min)
    b1.max = Math.max(b1.max, b2.max)
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