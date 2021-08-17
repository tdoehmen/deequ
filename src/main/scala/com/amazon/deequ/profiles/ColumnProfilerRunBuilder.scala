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

package com.amazon.deequ.profiles

import com.amazon.deequ.repository._
import com.amazon.deequ.analyzers.{DataTypeInstances, KLLParameters, KLLSketch}
import org.apache.spark.sql.{DataFrame, SparkSession}

/** A class to build a Constraint Suggestion run using a fluent API */
class ColumnProfilerRunBuilder(val data: DataFrame) {

  protected var printStatusUpdates: Boolean = false
  protected var cacheInputs: Boolean = false
  protected var lowCardinalityHistogramThreshold: Int =
    ColumnProfiler.DEFAULT_CARDINALITY_THRESHOLD
  protected var restrictToColumns: Option[Seq[String]] = None

  protected var metricsRepository: Option[MetricsRepository] = None
  protected var reuseExistingResultsKey: Option[ResultKey] = None
  protected var failIfResultsForReusingMissing: Boolean = false
  protected var saveOrAppendResultsKey: Option[ResultKey] = None

  protected var sparkSession: Option[SparkSession] = None
  protected var overwriteOutputFiles: Boolean = false
  protected var saveColumnProfilesJsonPath: Option[String] = None
  protected var saveConstraintSuggestionsJsonPath: Option[String] = None
  protected var saveEvaluationResultsJsonPath: Option[String] = None
  protected var correlation = false
  protected var histogram = false
  protected var kllProfiling = false
  protected var kllParameters: Option[KLLParameters] = None
  protected var predefinedTypes: Map[String, DataTypeInstances.Value] = Map.empty
  protected var maxCorrelationCols: Option[Int] = None
  protected var exactUniqueness = false
  protected var exactUniquenessCols: Option[Seq[String]] = None
  protected var optimized = true

  protected def this(constraintSuggestionRunBuilder: ColumnProfilerRunBuilder) {

    this(constraintSuggestionRunBuilder.data)

    printStatusUpdates = constraintSuggestionRunBuilder.printStatusUpdates
    cacheInputs = constraintSuggestionRunBuilder.cacheInputs
    lowCardinalityHistogramThreshold = constraintSuggestionRunBuilder
      .lowCardinalityHistogramThreshold
    restrictToColumns = constraintSuggestionRunBuilder.restrictToColumns

    metricsRepository = constraintSuggestionRunBuilder.metricsRepository
    reuseExistingResultsKey = constraintSuggestionRunBuilder.reuseExistingResultsKey
    failIfResultsForReusingMissing = constraintSuggestionRunBuilder.failIfResultsForReusingMissing
    saveOrAppendResultsKey = constraintSuggestionRunBuilder.saveOrAppendResultsKey

    sparkSession = constraintSuggestionRunBuilder.sparkSession
    overwriteOutputFiles = constraintSuggestionRunBuilder.overwriteOutputFiles
    saveColumnProfilesJsonPath = constraintSuggestionRunBuilder.saveColumnProfilesJsonPath
    saveConstraintSuggestionsJsonPath = constraintSuggestionRunBuilder
      .saveConstraintSuggestionsJsonPath
    saveEvaluationResultsJsonPath = constraintSuggestionRunBuilder.saveEvaluationResultsJsonPath

    restrictToColumns = constraintSuggestionRunBuilder.restrictToColumns
    correlation = constraintSuggestionRunBuilder.correlation
    maxCorrelationCols = constraintSuggestionRunBuilder.maxCorrelationCols
    histogram = constraintSuggestionRunBuilder.histogram

    kllProfiling = constraintSuggestionRunBuilder.kllProfiling
    kllParameters = constraintSuggestionRunBuilder.kllParameters
    predefinedTypes = constraintSuggestionRunBuilder.predefinedTypes
    exactUniqueness = constraintSuggestionRunBuilder.exactUniqueness
    exactUniquenessCols = constraintSuggestionRunBuilder.exactUniquenessCols
    optimized = constraintSuggestionRunBuilder.optimized
  }

  /**
    * Print status updates between passes
    *
    * @param printStatusUpdates Whether to print status updates
    */
  def printStatusUpdates(printStatusUpdates: Boolean): this.type = {
    this.printStatusUpdates = printStatusUpdates
    this
  }

  /**
    * Cache inputs
    *
    * @param cacheInputs Whether to cache inputs
    */
  def cacheInputs(cacheInputs: Boolean): this.type = {
    this.cacheInputs = cacheInputs
    this
  }

  /**
    * Set the thresholds of values until it is considered to expensive to
    * calculate the histograms (for backwards compatability)
    *
    * @param lowCardinalityHistogramThreshold The threshold
    */
  def withLowCardinalityHistogramThreshold(lowCardinalityHistogramThreshold: Int): this.type = {
    this.lowCardinalityHistogramThreshold = lowCardinalityHistogramThreshold
    this
  }

  /**
    * Can be used to specify a subset of columns to look at
    *
    * @param restrictToColumns The columns to look at
    */
  def restrictToColumns(restrictToColumns: Seq[String]): this.type = {
    this.restrictToColumns = Option(restrictToColumns)
    this
  }

  /**
   * Enable correlation profiling on Numerical columns, disabled by default.
   *
   * @param correlation Enable oder disable correlation profiling
   * @param maxCorrelationCols The maximum number of columns to calculate correlations on
   */
  def withCorrelation(correlation: Boolean, maxCorrelationCols: Int = 100): this.type = {
    this.correlation = correlation
    this.maxCorrelationCols = Some(maxCorrelationCols)
    this
  }

  /**
   * Enable histogram profiling on Numerical and Categorial columns, disabled by default.
   *
   * @param histogram Enable oder disable histogram profiling
   * @param maxBuckets The maximum number of distinct values to calculate the histogram for
   */
  def withHistogram(histogram: Boolean, maxBuckets: Int = 20): this.type = {
    this.histogram = histogram
    this.kllProfiling = histogram
    this.lowCardinalityHistogramThreshold = maxBuckets
    this.kllParameters = Some(KLLParameters(KLLSketch.DEFAULT_SKETCH_SIZE, KLLSketch
      .DEFAULT_SHRINKING_FACTOR, maxBuckets));
    this
  }

  /**
   * Enables exact Uniqueness, Entropy and Distinctness for all columns
   *
   * @param exactUniqueness Enable oder disable uniqueness, entropy and distinctness profiling
   */
  def withExactUniqueness(exactUniqueness: Boolean): this.type = {
    this.exactUniqueness = exactUniqueness
    this
  }

  /**
   * Enables exact Uniqueness, Entropy and Distinctness for specified columns
   *
   * @param exactUniquenessColumns List of columns that should be selected for uniqueness profiling
   */
  def restrictExactUniquenessColumns(exactUniquenessColumns: Seq[String]): this.type = {
    this.exactUniquenessCols = Some(exactUniquenessColumns)
    this
  }

  /**
   * Use unoptimized version of profiler (optimizations on by default)
   *
   */
  def nonOptimized(): this.type = {
    this.optimized = false
    this
  }

  /**
   * Enable KLL Sketches profiling on Numerical columns, disabled by default.
   * (for backwards compatability)
   */
  def withKLLProfiling(): this.type = {
    this.kllProfiling = true
    this
  }

  /**
   * Set KLL parameters.
   * (for backwards compatability)
   *
   * @param kllParameters kllParameters(sketchSize, shrinkingFactor, numberOfBuckets)
   */
  def setKLLParameters(kllParameters: Option[KLLParameters]): this.type = {
    this.kllParameters = kllParameters
    this
  }

  /**
   * Set predefined data types for each column (e.g. baseline)
   * (for backwards compatability)
   *
   * @param dataTypes dataType map for baseline columns
   */
  def setPredefinedTypes(dataTypes: Map[String, DataTypeInstances.Value]): this.type = {
    this.predefinedTypes = dataTypes
    this
  }


  /**
    * Set a metrics repository associated with the current data to enable features like reusing
    * previously computed results and storing the results of the current run.
    *
    * @param metricsRepository A metrics repository to store and load results associated with the
    *                          run
    */
  def useRepository(metricsRepository: MetricsRepository)
    : ColumnProfilerRunBuilderWithRepository = {

    new ColumnProfilerRunBuilderWithRepository(this, Option(metricsRepository))
  }

  /**
    * Use a sparkSession to conveniently create output files
    *
    * @param sparkSession The SparkSession
    */
  def useSparkSession(
      sparkSession: SparkSession)
    : ColumnProfilerRunBuilderWithSparkSession = {

    new ColumnProfilerRunBuilderWithSparkSession(this, Option(sparkSession))
  }

  def run(): ColumnProfiles = {
    ColumnProfilerRunner().run(
      data,
      restrictToColumns,
      lowCardinalityHistogramThreshold,
      printStatusUpdates,
      cacheInputs,
      ColumnProfilerRunBuilderFileOutputOptions(
        sparkSession,
        saveColumnProfilesJsonPath,
        overwriteOutputFiles),
      ColumnProfilerRunBuilderMetricsRepositoryOptions(
        metricsRepository,
        reuseExistingResultsKey,
        failIfResultsForReusingMissing,
        saveOrAppendResultsKey),
      correlation,
      histogram,
      kllProfiling,
      kllParameters,
      predefinedTypes,
      optimized,
      maxCorrelationCols,
      exactUniqueness,
      exactUniquenessCols
    )
  }
}

class ColumnProfilerRunBuilderWithRepository(
    columnProfilerRunBuilder: ColumnProfilerRunBuilder,
    usingMetricsRepository: Option[MetricsRepository])
  extends ColumnProfilerRunBuilder(columnProfilerRunBuilder) {

  metricsRepository = usingMetricsRepository

   /**
    * Reuse any previously computed results stored in the metrics repository associated with the
    * current data to save computation time.
    *
    * @param resultKey The exact result key of the previously computed result
    */
  def reuseExistingResultsForKey(
      resultKey: ResultKey,
      failIfResultsMissing: Boolean = false)
    : this.type = {

    reuseExistingResultsKey = Option(resultKey)
    failIfResultsForReusingMissing = failIfResultsMissing
    this
  }

  /**
    * A shortcut to save the results of the run or append them to existing results in the
    * metrics repository.
    *
    * @param resultKey The result key to identify the current run
    */
  def saveOrAppendResult(resultKey: ResultKey): this.type = {
    saveOrAppendResultsKey = Option(resultKey)
    this
  }
}

class ColumnProfilerRunBuilderWithSparkSession(
    columnProfilerRunBuilder: ColumnProfilerRunBuilder,
    usingSparkSession: Option[SparkSession])
  extends ColumnProfilerRunBuilder(columnProfilerRunBuilder) {

  sparkSession = usingSparkSession

  /**
    * Save the column profiles json to e.g. S3
    *
    * @param path The file path
    */
  def saveColumnProfilesJsonToPath(
      path: String)
    : this.type = {

    saveColumnProfilesJsonPath = Option(path)
    this
  }

  /**
    * Whether previous files with identical names should be overwritten when
    * saving files to some file system.
    *
    * @param overwriteFiles Whether previous files with identical names
    *                       should be overwritten
    */
  def overwritePreviousFiles(overwriteFiles: Boolean): this.type = {
    overwriteOutputFiles = overwriteOutputFiles
    this
  }
}
