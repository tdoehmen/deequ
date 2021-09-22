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
 * based on fast-mRMR (https://github.com/sramirez/fast-mRMR)
 */

package com.amazon.deequ.featureselection

import com.amazon.deequ.featureselection.{InfoTheory => IT}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import scala.collection.immutable.ListMap

/**
 * Minimum-Redundancy Maximum-Relevance criterion (mRMR)
 */
class MrmrCriterion(var relevance: Double) extends Serializable {

  var redundance: Double = 0.0
  var selectedSize: Int = 0

  def score = {
    if (selectedSize != 0) {
      relevance - redundance / selectedSize
    } else {
      relevance
    }
  }

  def update(mi: Double): MrmrCriterion = {
    redundance += mi
    selectedSize += 1
    this
  }

  override def toString: String = "MRMR"
}

/**
 * Train a info-theory feature selection model according to a criterion.
 */
class MrmrSelector protected extends Serializable {

  // Pool of criterions
  private type Pool = RDD[(Int, MrmrCriterion)]
  // Case class for criterions by feature
  protected case class F(feat: Int, crit: Double)

  /**
   * Perform a info-theory selection process.
   *
   * @param data Columnar data (last element is the class attribute).
   * @param nToSelect Number of features to select.
   * @param nFeatures Number of total features in the dataset.
   * @return A list with the most relevant features and its scores.
   *
   */
  private def selectFeatures(
                                       data: RDD[(Long, Byte)],
                                       nToSelect: Int,
                                       nFeatures: Int,
                                       verbose: Boolean) = {

    val label = nFeatures - 1
    val nInstances = data.count() / nFeatures
    val counterByKey = data.map({ case (k, v) => (k % nFeatures).toInt -> (v & 0xff)})
      .distinct()
      .groupByKey().mapValues(_.max + 1).collectAsMap().toMap

    // calculate relevance
    val MiAndCmi = IT.computeMI(
      data, 0 until label, label, nInstances, nFeatures, counterByKey)
    var pool = MiAndCmi.map{case (x, mi) => (x, new MrmrCriterion(mi))}
      .collectAsMap()
    if (verbose) {
      // Print most relevant features
      val strRels = MiAndCmi.collect().sortBy(-_._2)
        .take(nToSelect)
        .map({case (f, mi) => (f + 1) + "\t" + "%.4f" format mi})
        .mkString("\n")
      println("\n*** MaxRel features ***\nFeature\tScore\n" + strRels)
    }
    // get maximum and select it
    val firstMax = pool.maxBy(_._2.score)
    var selected = Seq(F(firstMax._1, firstMax._2.score))
    pool = pool - firstMax._1

    while (selected.size < nToSelect) {
      // update pool
      val newMiAndCmi = IT.computeMI(data, pool.keys.toSeq,
        selected.head.feat, nInstances, nFeatures, counterByKey)
        .map({ case (x, crit) => (x, crit) })
        .collectAsMap()

      pool.foreach({ case (k, crit) =>
        newMiAndCmi.get(k) match {
          case Some(_) => crit.update(_)
          case None =>
        }
      })

      // get maximum and save it
      // TODO: takes lowest feature index if scores are equal
      val max = pool.maxBy(_._2.score)
      // select the best feature and remove from the whole set of features
      selected = F(max._1, max._2.score) +: selected
      pool = pool - max._1
    }
    selected.reverse
  }

  private def runColumnar(
                            columnarData: RDD[(Long, Byte)],
                            nToSelect: Int,
                            nAllFeatures: Int,
                            verbose: Boolean = true): Seq[(Int, Double)] = {
    columnarData.persist(StorageLevel.MEMORY_AND_DISK_SER)

    require(nToSelect < nAllFeatures)
    val selected = selectFeatures(columnarData, nToSelect, nAllFeatures, verbose)

    columnarData.unpersist()

    selected.map{case F(feat, rel) => feat -> rel}
  }
}

object MrmrSelector {

  /**
   * Train a mRMR selection model according to a given criterion
   * and return a subset of data.
   *
   * @param   data RDD of LabeledPoint (discrete data as integers in range [0, 255]).
   * @param   nToSelect maximum number of features to select
   * @param   nAllFeatures number of features to select.
   * @param   indexToFeatures map of df-indexes and corresponding column names.
   * @param   verbose whether messages should be printed or not.
   * @return  A mRMR selector that selects a subset of features from the original dataset.
   *
   * Note: LabeledPoint data must be integer values in double representation 
   * with a maximum of 256 distinct values. In this manner, data can be transformed
   * to byte class directly, making the selection process much more efficient. 
   *
   */
  def selectFeatures(
             data: RDD[(Long, Byte)],
             nToSelect: Int = -1,
             nAllFeatures: Int,
             indexToFeatures: Map[Int, String],
             verbose: Boolean = false): Map[String, Double] = {
    // if nToSelect -1 or larger than nAllFeatures, clamp to nAllFeatures-1
    val nSelect = if(nToSelect < 0 || nToSelect > nAllFeatures-1) nAllFeatures-1 else nToSelect

    val selected = new MrmrSelector().runColumnar(data, nSelect, nAllFeatures, verbose)

    if (verbose) {
      // Print best features according to the mRMR measure
      val out = selected.map { case (feat, rel) => (indexToFeatures(feat)) + "\t" + "%.4f"
        .format(rel) }.mkString("\n")
      println("\n*** mRMR features ***\nFeature\tScore\n" + out)
    }

    // Return best features and mRMR measure
    ListMap(selected.map( kv => indexToFeatures(kv._1) -> kv._2 ): _*)
  }

}

