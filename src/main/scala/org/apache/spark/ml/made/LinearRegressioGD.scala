
package org.apache.spark.ml.made

import scala.util.control.Breaks._
import breeze.linalg
import breeze.linalg.{DenseVector => BDV, Vector => BV}
import org.apache.spark.internal.Logging
import org.apache.spark.ml.{PredictorParams}
import org.apache.spark.ml.param.{DoubleParam, ParamMap, ParamValidators, Params}
import org.apache.spark.ml.regression.{RegressionModel, Regressor}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable, MLWritable, MetadataUtils}
import org.apache.spark.sql.types.{DataType, StructType}
import org.apache.spark.ml.linalg.{DenseVector, Vector}
import org.apache.spark.ml.param.shared.{HasMaxIter, HasTol}
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.{Dataset, Encoder, Row}
import org.apache.spark.mllib


trait HasLR extends Params {
  final val lr: DoubleParam = new DoubleParam(this, "lr", "gd learning rate", ParamValidators.gtEq(0))
  final def getLR: Double = $(lr)
}

trait LinearRegressionParams extends PredictorParams with HasLR with HasMaxIter {
  override protected def validateAndTransformSchema(
                                                     schema: StructType,
                                                     fitting: Boolean,
                                                     featuresDataType: DataType): StructType = {
    super.validateAndTransformSchema(schema, fitting, featuresDataType)
  }

  setDefault(lr -> 1e-4, maxIter -> 1000, tol -> 1e-5)
}

class LinearRegressionGD(override val uid: String)
  extends Regressor[Vector, LinearRegressionGD, LinearRegressionModel]
    with LinearRegressionParams with DefaultParamsWritable with Logging {

  def this() = this(Identifiable.randomUID("linRegGD"))

  def setLR(value: Double): this.type = set(lr, value)

  def setMaxIter(value: Int): this.type = set(maxIter, value)

  def setTol(value: Double): this.type = set(tol, value)

  override def copy(extra: ParamMap): LinearRegressionGD = defaultCopy(extra)

  override protected def train(dataset: Dataset[_]): LinearRegressionModel = {
    val numFeatures = MetadataUtils.getNumFeatures(dataset, $(featuresCol))

    // Used to convert untyped dataframes to datasets with vectors
    implicit val vectorEncoder : Encoder[Vector] = ExpressionEncoder()
    implicit val doubleEncoder : Encoder[Double] = ExpressionEncoder()

    // Current coefficients
    var coefficients: BDV[Double] = BDV.ones[Double](numFeatures)
    var intercept: Double = 1.0
    var error: Double = Double.MaxValue

    // Convert input dataset
    val vectors: Dataset[(Vector, Double)] = dataset.select(
      dataset($(featuresCol)).as[Vector],
      dataset($(labelCol)).as[Double]
    )

    // Main loop
    for (i <- 1 to getMaxIter) {
      // Calculate coefficientsSummary over data
      val (coefficientsSummary, interceptSummary) = vectors.rdd.mapPartitions((data: Iterator[(Vector, Double)]) => {
        val coefficientsSummarizer = new MultivariateOnlineSummarizer()
        val residualSummarizer = new MultivariateOnlineSummarizer()

        data.foreach(r => {
          val x: linalg.Vector[Double] = r._1.asBreeze
          val y: Double = r._2
          val yhat: Double = (x dot coefficients) + intercept
          val residual: Double = y - yhat

          coefficientsSummarizer.add(mllib.linalg.Vectors.fromBreeze(x * residual))
          residualSummarizer.add(mllib.linalg.Vectors.dense(residual))
        })

        Iterator((coefficientsSummarizer, residualSummarizer))
      }).reduce((x, y) => {
        (x._1 merge y._1, x._2 merge y._2)
      })

      // Calculate error
      error = interceptSummary.mean(0)

      // Update coefficients
      var dCoeff: BDV[Double] = coefficientsSummary.mean.asBreeze.toDenseVector
      dCoeff :*= (-2.0) * getLR
      coefficients -= dCoeff

      // Update intercept
      var dInter = (-2.0) * getLR * error
      intercept -= dInter
    } }

    // Return fitted model
    val lrModel = copyValues(new LinearRegressionModel(uid, new DenseVector(coefficients.toArray), intercept))
    lrModel
  }
}

class LinearRegressionModel private[made](
                                           override val uid: String,
                                           val coefficients: Vector,
                                           val intercept: Double)
  extends RegressionModel[Vector, LinearRegressionModel]
    with LinearRegressionParams{

  val brzCoefficients: BV[Double] = coefficients.asBreeze

  private[made] def this(coefficients: Vector, intercept: Double) = this(Identifiable.randomUID("linRegGD"), coefficients.toDense, intercept)

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(new LinearRegressionModel(coefficients, intercept))

  override def predict(features: Vector): Double = {
    (features.asBreeze dot brzCoefficients) + intercept
  }
}