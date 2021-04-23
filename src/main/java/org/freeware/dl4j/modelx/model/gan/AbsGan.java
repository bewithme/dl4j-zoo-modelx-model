package org.freeware.dl4j.modelx.model.gan;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.graph.ReshapeVertex;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.zoo.ZooModel;
import org.freeware.dl4j.nn.GraphLayerItem;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

@Slf4j
public abstract class AbsGan extends ZooModel {


    /**
     * 从对抗网络中把参数复制给生成器和判别器
     * @param generator
     * @param discriminator
     * @param gan
     */
    public void copyParamsFromGanToGeneratorAndDiscriminator(ComputationGraph generator,ComputationGraph discriminator, ComputationGraph gan) {
       copyParamsFromGanToGenerator(generator,gan);
       copyParamsFromGanToDiscriminator(discriminator,gan);
    }




    /**
     * 从对抗网络中把参数复制给生成器
     * @param generator
     * @param gan
     */
    public void copyParamsFromGanToGenerator(ComputationGraph generator, ComputationGraph gan) {
        int genLayerLen = generator.getLayers().length;
        for (int i = 0; i < genLayerLen; i++) {
            generator.getLayer(i).setParams(gan.getLayer(i).params());
        }
    }

    /**
     * 从对抗网络中把参数复制给判别器
     * @param discriminator
     * @param gan
     */
    public void copyParamsFromGanToDiscriminator(ComputationGraph discriminator, ComputationGraph gan) {

        int ganLayerLen =gan.getLayers().length;

        int disLayerLen=discriminator.getLayers().length;

        int ganLayerIndex = ganLayerLen-disLayerLen;

        for (int disLayerIndex =0; disLayerIndex < disLayerLen; disLayerIndex++) {

            INDArray params= gan.getLayer(ganLayerIndex).params();

            ganLayerIndex++;

            discriminator.getLayer(disLayerIndex).setParams(params);
        }
    }

    /**
     * 从判别器复制参数到对抗网络的判别器中
     * discriminator, gan
     * @param discriminator
     * @param gan
     */
    public  void copyParamsFromDiscriminatorToGanDiscriminator(ComputationGraph discriminator, ComputationGraph gan) {

        int disLayerLen=discriminator.getLayers().length;

        int ganLayerIndex= gan.getLayers().length-disLayerLen;

        for (int disLayerIndex = 0; disLayerIndex < disLayerLen; disLayerIndex++) {

            INDArray params=discriminator.getLayer(disLayerIndex ).params();

            gan.getLayer(ganLayerIndex).setParams(params);

            ganLayerIndex++;
        }
    }


    /**
     * 批量添加计算图的层或顶点
     * @param graphBuilder
     * @param graphLayerItems
     * @param frozen
     */
    protected void addGraphItems(ComputationGraphConfiguration.GraphBuilder graphBuilder, List<GraphLayerItem> graphLayerItems, Boolean frozen) {

        for(GraphLayerItem graphLayerItem:graphLayerItems) {

            if(graphLayerItem.getLayerOrVertex() instanceof MergeVertex) {

                MergeVertex mergeVertex=(MergeVertex)graphLayerItem.getLayerOrVertex();

                graphBuilder.addVertex(graphLayerItem.getLayerName(), mergeVertex, graphLayerItem.getLayerInputs());

            } else if(graphLayerItem.getLayerOrVertex() instanceof ElementWiseVertex) {

                ElementWiseVertex elementWiseVertex=(ElementWiseVertex)graphLayerItem.getLayerOrVertex();

                graphBuilder.addVertex(graphLayerItem.getLayerName(), elementWiseVertex, graphLayerItem.getLayerInputs());

            }else if (graphLayerItem.getLayerOrVertex() instanceof ReshapeVertex){

                ReshapeVertex reshapeVertex=(ReshapeVertex)graphLayerItem.getLayerOrVertex();

                graphBuilder.addVertex(graphLayerItem.getLayerName(), reshapeVertex, graphLayerItem.getLayerInputs());

            }else if (graphLayerItem.getLayerOrVertex() instanceof Layer){

                Layer layer=(Layer)graphLayerItem.getLayerOrVertex();

                graphBuilder.addLayer(graphLayerItem.getLayerName(), layer, graphLayerItem.getLayerInputs());

            }
        }
    }






}
