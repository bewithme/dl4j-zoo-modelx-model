package org.freeware.dl4j.modelx.model.gan;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.freeware.dl4j.modelx.model.ZooModelX;
import org.nd4j.linalg.api.ndarray.INDArray;


@Slf4j
public abstract class AbsGan extends ZooModelX {



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

    public void copyParamsFromGanToGeneratorByName(ComputationGraph generator, ComputationGraph gan) {

        org.deeplearning4j.nn.api.Layer[] generatorLayers= generator.getLayers();

        for(org.deeplearning4j.nn.api.Layer generatorLayer:generatorLayers){

            String generatorLayerName=generatorLayer.getConfig().getLayerName();

            generatorLayer.setParams( gan.getLayer(generatorLayerName).params());
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
     * 通过层名称来复制参数
     * @param discriminator
     * @param gan
     */
    public  void copyParamsFromDiscriminatorToGanDiscriminatorByName(ComputationGraph discriminator, ComputationGraph gan) {

        org.deeplearning4j.nn.api.Layer[] discriminatorLayers=discriminator.getLayers();

        for(org.deeplearning4j.nn.api.Layer discriminatorLayer:discriminatorLayers){

           String discriminatorLayerName=discriminatorLayer.getConfig().getLayerName();

           gan.getLayer(discriminatorLayerName).setParams(discriminatorLayer.params());

        }
    }












}
