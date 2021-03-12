package org.freeware.dl4j.modelx.model.yolo;

import lombok.Data;
import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.objdetect.BoundingBoxesDeserializer;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.params.EmptyParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.regularization.Regularization;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.impl.LossBinaryXENT;
import org.nd4j.linalg.lossfunctions.impl.LossL2;
import org.nd4j.serde.jackson.shaded.NDArrayTextSerializer;
import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;
import org.nd4j.shade.jackson.databind.annotation.JsonSerialize;

import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Map;
@Data
public class Yolo3OutputLayerConfiguration extends org.deeplearning4j.nn.conf.layers.Layer {

    private double lambdaCoord;
    private double lambdaNoObj;
    private ILossFunction lossPositionScale;
    private ILossFunction lossClassPredictions;
    @JsonSerialize(using = NDArrayTextSerializer.class)
    @JsonDeserialize(using = BoundingBoxesDeserializer.class)
    private INDArray boundingBoxPriors;

    private Yolo3OutputLayerConfiguration() {
        //No-arg constructor for Jackson JSON
    }

    private Yolo3OutputLayerConfiguration(Yolo3OutputLayerConfiguration.Builder builder) {
        super(builder);
        this.lambdaCoord = builder.lambdaCoord;
        this.lambdaNoObj = builder.lambdaNoObj;
        this.lossPositionScale = builder.lossPositionScale;
        this.lossClassPredictions = builder.lossClassPredictions;
        this.boundingBoxPriors = builder.boundingBoxPriors;
    }

    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> trainingListeners,
                             int layerIndex, INDArray layerParamsView, boolean initializeParams, DataType networkDataType) {
        Yolo3OutputLayer ret = new Yolo3OutputLayer(conf, networkDataType);
        ret.setListeners(trainingListeners);
        ret.setIndex(layerIndex);
        ret.setParamsViewArray(layerParamsView);
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(conf);
        return ret;
    }

    @Override
    public ParamInitializer initializer() {
        return EmptyParamInitializer.getInstance();
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        return inputType; //Same shape output as input
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        //No op
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        switch (inputType.getType()) {
            case FF:
            case RNN:
                throw new UnsupportedOperationException("Cannot use FF or RNN input types");
            case CNN:
                return null;
            case CNNFlat:
                InputType.InputTypeConvolutionalFlat cf = (InputType.InputTypeConvolutionalFlat) inputType;
                return new FeedForwardToCnnPreProcessor(cf.getHeight(), cf.getWidth(), cf.getDepth());
            default:
                return null;
        }
    }

    @Override
    public List<Regularization> getRegularizationByParam(String paramName) {
        //Not applicable
        return null;
    }

    @Override
    public boolean isPretrainParam(String paramName) {
        return false; //No params
    }

    @Override
    public GradientNormalization getGradientNormalization() {
        return GradientNormalization.None;
    }

    @Override
    public double getGradientNormalizationThreshold() {
        return 1.0;
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        long numValues = inputType.arrayElementsPerExample();

        //This is a VERY rough estimate...
        return new LayerMemoryReport.Builder(layerName, Yolo2OutputLayer.class, inputType, inputType)
                .standardMemory(0, 0) //No params
                .workingMemory(0, numValues, 0, 6 * numValues).cacheMemory(0, 0) //No cache
                .build();
    }

    @Getter
    @Setter
    public static class Builder extends org.deeplearning4j.nn.conf.layers.Layer.Builder<Yolo2OutputLayer.Builder> {

        /**
         * Loss function coefficient for position and size/scale components of the loss function. Default (as per
         * paper): 5
         *
         */
        private double lambdaCoord = 5;

        /**
         * Loss function coefficient for the "no object confidence" components of the loss function. Default (as per
         * paper): 0.5
         *
         */
        private double lambdaNoObj = 0.5;

        /**
         * Loss function for position/scale component of the loss function
         *
         */
        private ILossFunction lossPositionScale = new LossL2();

        /**
         * Loss function for the class predictions - defaults to L2 loss (i.e., sum of squared errors, as per the
         * paper), however Loss MCXENT could also be used (which is more common for classification).
         *
         */
        private ILossFunction lossClassPredictions =new LossBinaryXENT();

        /**
         * Bounding box priors dimensions [width, height]. For N bounding boxes, input has shape [rows, columns] = [N,
         * 2] Note that dimensions should be specified as fraction of grid size. For example, a network with 13x13
         * output, a value of 1.0 would correspond to one grid cell; a value of 13 would correspond to the entire
         * image.
         *
         */
        private INDArray boundingBoxPriors;

        /**
         * Loss function coefficient for position and size/scale components of the loss function. Default (as per
         * paper): 5
         *
         * @param lambdaCoord Lambda value for size/scale component of loss function
         */
        public Yolo3OutputLayerConfiguration.Builder lambdaCoord(double lambdaCoord) {
            this.setLambdaCoord(lambdaCoord);
            return this;
        }

        /**
         * Loss function coefficient for the "no object confidence" components of the loss function. Default (as per
         * paper): 0.5
         *
         * @param lambdaNoObj Lambda value for no-object (confidence) component of the loss function
         */
        public Yolo3OutputLayerConfiguration.Builder lambdaNoObj(double lambdaNoObj) {
            this.setLambdaNoObj(lambdaNoObj);
            return this;
        }

        /**
         * Loss function for position/scale component of the loss function
         *
         * @param lossPositionScale Loss function for position/scale
         */
        public Yolo3OutputLayerConfiguration.Builder lossPositionScale(ILossFunction lossPositionScale) {
            this.setLossPositionScale(lossPositionScale);
            return this;
        }

        /**
         * Loss function for the class predictions - defaults to L2 loss (i.e., sum of squared errors, as per the
         * paper), however Loss MCXENT could also be used (which is more common for classification).
         *
         * @param lossClassPredictions Loss function for the class prediction error component of the YOLO loss function
         */
        public Yolo3OutputLayerConfiguration.Builder lossClassPredictions(ILossFunction lossClassPredictions) {
            this.setLossClassPredictions(lossClassPredictions);
            return this;
        }

        /**
         * Bounding box priors dimensions [width, height]. For N bounding boxes, input has shape [rows, columns] = [N,
         * 2] Note that dimensions should be specified as fraction of grid size. For example, a network with 13x13
         * output, a value of 1.0 would correspond to one grid cell; a value of 13 would correspond to the entire
         * image.
         *
         * @param boundingBoxPriors Bounding box prior dimensions (width, height)
         */
        public Yolo3OutputLayerConfiguration.Builder boundingBoxPriors(INDArray boundingBoxPriors) {
            this.setBoundingBoxPriors(boundingBoxPriors);
            return this;
        }

        @Override
        public Yolo3OutputLayerConfiguration build() {
            if (boundingBoxPriors == null) {
                throw new IllegalStateException("Bounding boxes have not been set");
            }

            if (boundingBoxPriors.rank() != 2 || boundingBoxPriors.size(1) != 2) {
                throw new IllegalStateException("Bounding box priors must have shape [nBoxes, 2]. Has shape: "
                        + Arrays.toString(boundingBoxPriors.shape()));
            }

            return new Yolo3OutputLayerConfiguration(this);
        }
    }
}

