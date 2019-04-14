package com.amazon.deequ.constraints

/**
  * Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import com.amazon.deequ.metrics.Metric
import com.amazon.deequ.statistics.Statistic

import scala.util.{Failure, Success}

/**
  * Common trait for all statistics based constraints that provides unified way to access
  * ComputedStatistics and metrics stored in it.
  *
  * Runs the analysis and get the value of the metric returned by the statistic,
  * picks the numeric value that will be used in the assertion function with metric picker
  * runs the assertion.
  *
  * @param statistic
  * @param assertion   Assertion function
  * @param valuePicker Optional function to pick the interested part of the metric value that the
  *                    assertion will be running on. Absence of such function means the metric
  *                    value would be used in the assertion as it is.
  * @param hint A hint to provide additional context why a constraint could have failed
  * @tparam M : Type of the metric value
  * @tparam V : Type of the value being used in assertion function
  *
  */
private[deequ] case class StatisticConstraint[M, V](
    statistic: Statistic,
    private[deequ] val assertion: V => Boolean,
    private[deequ] val valuePicker: Option[M => V] = None,
    private[deequ] val hint: Option[String] = None,
    private[deequ] val name: Option[String] = None)
  extends Constraint {

  override def toString: String = {
    name match {
      case Some(str) => str
      case _  => s"StatisticContraint($statistic,$hint)"
    }
  }

  override def evaluate(
      analysisResults: Map[Statistic, Metric[_]])
    : ConstraintResult = {

    val metric = analysisResults.get(statistic).map(_.asInstanceOf[Metric[M]])

    metric.map(pickValueAndAssert).getOrElse(
      // Analysis is missing
      ConstraintResult(this, ConstraintStatus.Failure,
        Some(StatisticConstraint.MissingAnalysis), metric)
    )
  }

  private[this] def pickValueAndAssert(metric: Metric[M]): ConstraintResult = {

    metric.value match {
      // Analysis done successfully and result metric is there
      case Success(metricValue) =>
        try {
          val assertOn = runPickerOnMetric(metricValue)
          val assertionOk = runAssertion(assertOn)

          if (assertionOk) {
            ConstraintResult(this, ConstraintStatus.Success, metric = Some(metric))
          } else {
            var errorMessage = s"Value: $assertOn does not meet the constraint requirement!"
            hint.foreach(hint => errorMessage += s" $hint")

            ConstraintResult(this, ConstraintStatus.Failure, Some(errorMessage), Some(metric))
          }

        } catch {
          case StatisticConstraint.ConstraintAssertionException(msg) =>
            ConstraintResult(this, ConstraintStatus.Failure,
              Some(s"${StatisticConstraint.AssertionException}: $msg!"), Some(metric))
          case StatisticConstraint.ValuePickerException(msg) =>
            ConstraintResult(this, ConstraintStatus.Failure,
              Some(s"${StatisticConstraint.ProblematicMetricPicker}: $msg!"), Some(metric))
        }
      // An exception occurred during analysis
      case Failure(e) => ConstraintResult(this,
        ConstraintStatus.Failure, Some(e.getMessage), Some(metric))
    }
  }

  private def runPickerOnMetric(metricValue: M): V =
    try {
      valuePicker.map(function => function(metricValue)).getOrElse(metricValue.asInstanceOf[V])
    } catch {
      case e: Exception => throw StatisticConstraint.ValuePickerException(e.getMessage)
    }

  private def runAssertion(assertOn: V): Boolean =
    try {
      assertion(assertOn)
    } catch {
      case e: Exception => throw StatisticConstraint.ConstraintAssertionException(e.getMessage)
    }

}

private[deequ] object StatisticConstraint {
  val MissingAnalysis = "Missing Analysis, can't run the constraint!"
  val ProblematicMetricPicker = "Can't retrieve the value to assert on"
  val AssertionException = "Can't execute the assertion"

  private case class ValuePickerException(message: String) extends RuntimeException(message)
  private case class ConstraintAssertionException(message: String) extends RuntimeException(message)
}
