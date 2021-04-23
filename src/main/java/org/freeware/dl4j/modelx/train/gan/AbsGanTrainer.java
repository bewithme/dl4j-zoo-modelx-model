package org.freeware.dl4j.modelx.train.gan;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.freeware.dl4j.modelx.utils.INDArrayUtils;
import org.freeware.dl4j.modelx.utils.RandomUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Random;

@Slf4j
public abstract class AbsGanTrainer {


    protected static void setListeners(ComputationGraph discriminator, ComputationGraph gan) {

        UIServer uiServer = UIServer.getInstance();

        StatsStorage statsStorage = new InMemoryStatsStorage();

        uiServer.attach(statsStorage);

        discriminator.setListeners(new PerformanceListener(10, true),new StatsListener(statsStorage));

        gan.setListeners(new PerformanceListener(10, true),new StatsListener(statsStorage));
    }



    /**
     * 训练判别器
     * @param generator
     * @param discriminator
     * @param realFeature
     * @param realLabel
     *
     */
    protected static void trainDiscriminator(ComputationGraph generator, ComputationGraph discriminator, INDArray realFeature, INDArray realLabel, int batchSize,int latentDimLen, int labelIdxStart, int labelIdxEnd, Random random) {

        GeneratorInput generatorInput=getGeneratorInput( batchSize, latentDimLen,  labelIdxStart,  labelIdxEnd,  random);

        //创建batchSize行，100列的随机数浅层空间
        INDArray latentDim =generatorInput.getLatentDim();

        INDArray fakeLabel=generatorInput.getEmbeddingLabel();
        //用生成器生成假图片
        INDArray fakeImaged=generator.output(latentDim,fakeLabel)[0];

        INDArray[] fakeFeatures=new INDArray[] {fakeImaged,fakeLabel};

        INDArray[] fakeDisLabels=new INDArray[] {Nd4j.zeros(batchSize, 1)};

        MultiDataSet fakeMultiDataSet=new MultiDataSet(fakeFeatures,fakeDisLabels);

        realLabel= INDArrayUtils.toEmbeddingFormat(realLabel);

        INDArray[] realFeatures=new INDArray[] {realFeature,realLabel};

        INDArray[] realDisLabels=new INDArray[] {Nd4j.ones(batchSize, 1)};
        //构建多数据集（多个特征，多个标签）
        MultiDataSet realMultiDataSet=new MultiDataSet(realFeatures,realDisLabels);
        //用真实数据训练判别器
        discriminator.fit(realMultiDataSet);
        //用假数据训练判别器
        discriminator.fit(fakeMultiDataSet);
    }

    protected static void trainDiscriminator(ComputationGraph generator, ComputationGraph discriminator, INDArray realFeature, INDArray realLabel,int batchSize) {

        //用生成器生成假图片
        INDArray fakeImaged=generator.output(realFeature)[0];

        log.info(fakeImaged.shapeInfoToString());

        log.info(realFeature.shapeInfoToString());

        INDArray[] fakeFeatures=new INDArray[] {fakeImaged};

        INDArray[] fakeDisLabels=new INDArray[] {Nd4j.zeros(batchSize, 1)};

        MultiDataSet fakeMultiDataSet=new MultiDataSet(fakeFeatures,fakeDisLabels);

        INDArray[] realFeatures=new INDArray[] {realLabel};

        INDArray[] realDisLabels=new INDArray[] {Nd4j.ones(batchSize, 1)};
        //构建多数据集（多个特征，多个标签）
        MultiDataSet realMultiDataSet=new MultiDataSet(realFeatures,realDisLabels);
        //用真实数据训练判别器
        discriminator.fit(realMultiDataSet);
        //用假数据训练判别器
        discriminator.fit(fakeMultiDataSet);
    }


    /**
     * 对抗训练
     * 此时判别器的学习率为0
     * 所以只会训练生成器
     * @param gan
     * @param batchSize
     *
     */
    protected static void trainGan(ComputationGraph gan,int batchSize,int latentDimLen, int labelIdxStart, int labelIdxEnd, Random random) {

        GeneratorInput generatorInput=getGeneratorInput( batchSize, latentDimLen,  labelIdxStart,  labelIdxEnd,  random);
        //噪音数据
        INDArray noiseLatentDim = generatorInput.getLatentDim();
        //随机长成EmbeddingLayer输入格式的标签
        INDArray embeddingLabel = generatorInput.getEmbeddingLabel();
        //噪音特征
        INDArray[] noiseLatentFeature = new INDArray[]{noiseLatentDim, embeddingLabel};
        //这里故意把噪音的标签设为真，
        INDArray[] noiseLatentLabel = new INDArray[]{Nd4j.ones(batchSize, 1)};
        //对抗网络输入多数据集
        MultiDataSet ganInputMultiDataSet = new MultiDataSet(noiseLatentFeature, noiseLatentLabel);

        gan.fit(ganInputMultiDataSet);

    }

    /**
     * 对抗训练
     * 此时判别器的学习率为0
     * 所以只会训练生成器
     * @param gan
     * @param batchSize
     *
     */
    protected static void trainGan(ComputationGraph gan,int batchSize, INDArray realFeature) {

        //噪音特征
        INDArray[] noiseLatentFeature = new INDArray[]{realFeature};
        //这里故意把噪音的标签设为真，
        INDArray[] noiseLatentLabel = new INDArray[]{Nd4j.ones(batchSize, 1)};
        //对抗网络输入多数据集
        MultiDataSet ganInputMultiDataSet = new MultiDataSet(noiseLatentFeature, noiseLatentLabel);

        gan.fit(ganInputMultiDataSet);

    }

    /**
     * 获取生成器输入
     * @param batchSize
     * @param latentDimLen
     * @param labelIdxStart
     * @param labelIdxEnd
     * @param random
     * @return
     */
   protected static GeneratorInput getGeneratorInput(int batchSize, int latentDimLen, int labelIdxStart, int labelIdxEnd, Random random){
        //噪音数据
       INDArray latentDim = Nd4j.rand(new int[]{batchSize, latentDimLen});
       //随机长成EmbeddingLayer输入格式的标签
       INDArray embeddingLabel = RandomUtils.getRandomEmbeddingLabel(batchSize,labelIdxStart,labelIdxEnd,random);

       return new GeneratorInput(latentDim,embeddingLabel);
   }

   @AllArgsConstructor
   @Data
   static class GeneratorInput{
       private INDArray latentDim;
       private INDArray embeddingLabel;
   }

}
