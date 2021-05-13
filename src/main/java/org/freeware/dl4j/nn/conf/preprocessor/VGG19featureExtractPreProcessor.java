package org.freeware.dl4j.nn.conf.preprocessor;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.conf.CNN2DFormat;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.freeware.dl4j.modelx.utils.ImageUtils;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import java.io.IOException;


@Slf4j
@Data
public class VGG19featureExtractPreProcessor implements InputPreProcessor {

    protected long inputHeight;
    protected long inputWidth;
    protected long numChannels;
    protected CNN2DFormat format = CNN2DFormat.NCHW;    //Default for legacy JSON deserialization

    private ComputationGraph vgg19;

    public VGG19featureExtractPreProcessor(long inputHeight, long inputWidth, long numChannels, ComputationGraph vgg19) {
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        this.numChannels = numChannels;
        this.vgg19=vgg19;
    }


    @Override
    public INDArray preProcess(INDArray input, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {

        MemoryWorkspace workspace = Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("xxxx");
        try {
            input= ImageUtils.resize(input,224,224);
        } catch (IOException e) {
            e.printStackTrace();
        }
        input = workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, input);



        INDArray output=this.vgg19.output(false,workspace,input)[0];



        log.info(output.shapeInfoToString());

        return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, output);
    }

    @Override
    public INDArray backprop(INDArray epsilons, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {


        return workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, Nd4j.zeros(epsilons.size(0), numChannels, inputHeight, inputWidth)); //Move if required to specified workspace
    }

    @Override
    public InputPreProcessor clone() {
    try{
        VGG19featureExtractPreProcessor vgg19featureExtractPreProcessor=(VGG19featureExtractPreProcessor)super.clone();

        return vgg19featureExtractPreProcessor;
    } catch (CloneNotSupportedException e) {
        throw new RuntimeException(e);
    }


    }

    @Override
    public InputType getOutputType(InputType inputType) {
        if (inputType == null || inputType.getType() != InputType.Type.CNN) {
            throw new IllegalStateException("Invalid input type: Expected input of type CNN, got " + inputType);
        }

        return InputType.feedForward(4096);
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState, int minibatchSize) {
        return null;
    }
}
