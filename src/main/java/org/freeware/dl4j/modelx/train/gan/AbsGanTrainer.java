package org.freeware.dl4j.modelx.train.gan;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.freeware.dl4j.modelx.train.AbsTrainer;
import org.freeware.dl4j.modelx.utils.INDArrayUtils;
import org.freeware.dl4j.modelx.utils.RandomUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Random;

@Slf4j
public abstract class AbsGanTrainer extends AbsTrainer {





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

    protected static void trainDiscriminator(ComputationGraph generator, ComputationGraph discriminator, INDArray generatorInput, INDArray realFeature,int batchSize) {

        trainDiscriminator( generator,  discriminator,  generatorInput,  realFeature,new long[]{batchSize, 1});
    }


    protected static void trainDiscriminator(ComputationGraph generator, ComputationGraph discriminator, INDArray generatorInput, INDArray realFeature,long[] discriminatorOutputShape) {
        //用生成器生成假图片
        INDArray fakeImaged=generator.output(generatorInput)[0];

        INDArray[] fakeFeatures=new INDArray[] {fakeImaged};

        INDArray[] fakeDisLabels=new INDArray[] {Nd4j.zeros(discriminatorOutputShape)};

        MultiDataSet fakeMultiDataSet=new MultiDataSet(fakeFeatures,fakeDisLabels);

        INDArray[] realFeatures=new INDArray[] {realFeature};

        INDArray[] realDisLabels=new INDArray[] {Nd4j.ones(discriminatorOutputShape)};
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
    protected static void trainGan(ComputationGraph gan,int batchSize, INDArray generatorInput) {

        trainGan(gan,generatorInput,new long[]{batchSize, 1});

    }


    protected static void trainGan(ComputationGraph gan,INDArray generatorInput,long[] discriminatorOutputShape) {
        //噪音特征
        INDArray[] noiseLatentFeature = new INDArray[]{generatorInput};
        //这里故意把噪音的标签设为真，
        INDArray[] noiseLatentLabel = new INDArray[]{Nd4j.ones(discriminatorOutputShape)};
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


    protected static void trainReconstructNetwork(ComputationGraph reconstructNetwork, INDArray feature){

        INDArray[] features=new INDArray[] {feature};

        INDArray[] labels=new INDArray[] {feature};
        //构建多数据集（多个特征，多个标签）
        MultiDataSet multiDataSet=new MultiDataSet(features,labels);
        //用真实数据训练判别器
        reconstructNetwork.fit(multiDataSet);
   }


    protected static void trainIdentityMappingNetwork(ComputationGraph identityMappingNetwork, INDArray feature){

        INDArray[] features=new INDArray[] {feature};

        INDArray[] labels=new INDArray[] {feature};
        //构建多数据集（多个特征，多个标签）
        MultiDataSet multiDataSet=new MultiDataSet(features,labels);
        //用真实数据训练判别器
        identityMappingNetwork.fit(multiDataSet);

    }



    @AllArgsConstructor
   @Data
   static class GeneratorInput{
       private INDArray latentDim;
       private INDArray embeddingLabel;
   }

}
