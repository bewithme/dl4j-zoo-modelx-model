package org.freeware.dl4j.modelx.train.gan;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.freeware.dl4j.modelx.model.gan.CGan;
import org.freeware.dl4j.modelx.utils.INDArrayUtils;
import org.freeware.dl4j.modelx.utils.RandomUtils;
import org.freeware.dl4j.modelx.utils.VisualisationUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import java.io.File;
import java.io.IOException;
import java.util.Random;


/**
 * @author wenfengxu
 * 条件生成对抗网络训练
 * 1、把对抗网络的参数复制到生成器和判别器。
 * 2、用真实数据和生成器生成的假数据合并起来训练判别器。
 * 3、把判别器参数复制给对抗网络中的判别器，并冻结对抗网络中的判别器的参数使期不能学习。
 * 4、训练对抗网络，更新对抗网络中的生成器参数，然后把对抗网络中的生成器参数复制给生成器。
 */
@Slf4j
public class CGanTrainer {


    private  static Random random=new Random(12345);

    public static void main(String[] args) {

        CGan cgan= CGan.builder().build();

        ComputationGraph generator=cgan.initGenerator();

        ComputationGraph discriminator=cgan.initDiscriminator();

        ComputationGraph gan=cgan.init();

        generator.setListeners(new PerformanceListener(10, true));

        discriminator.setListeners(new PerformanceListener(10, true));

        cgan.copyParamsFromGanToGeneratorAndDiscriminator(generator,discriminator,gan);

        log.info(generator.summary());

        log.info(discriminator.summary());

        log.info(gan.summary());

        Nd4j.getMemoryManager().setAutoGcWindow(15 * 1000);

        MnistDataSetIterator trainData = null;
        try {
            trainData = new MnistDataSetIterator(128, true, 42);
        } catch (IOException e) {
            e.printStackTrace();
        }


        while (true) {

            trainData.reset();

            int iterationCounter = 0;


            while (trainData.hasNext()) {

                iterationCounter++;

                DataSet dataSet=trainData.next();

                INDArray realFeature = dataSet.getFeatures();

                INDArray realLabel = dataSet.getLabels();

                int batchSize = (int) realFeature.size(0);

                trainDiscriminator(generator, discriminator, realFeature, realLabel, batchSize);

                cgan.copyParamsFromDiscriminatorToGanDiscriminator(discriminator, gan);

                trainGan( gan, batchSize);

                cgan.copyParamsFromGanToGenerator(generator,gan);

                if (iterationCounter % 100 == 1) {

                    visualize(generator, batchSize,iterationCounter);

                }

            }

        }



    }

    /**
     * 测试数据可视化
     * @param generator 生成器
     *
     * @param batchSize 小批量大小
     * @param iterationCounter 迭代次数
     */
    private static void visualize(ComputationGraph generator, int batchSize,int iterationCounter) {

        INDArray[] testSamples = new INDArray[9];

        for(int k=0;k<9;k++){
            //创建batchSize行，100列的随机数浅层空间
            INDArray testLatentDim = Nd4j.rand(new int[]{batchSize,  100});

            INDArray label= RandomUtils.getRandomEmbeddingLabel(batchSize,0,9,random);

            INDArray testFakeImaged=generator.output(testLatentDim,label)[0];

            testSamples[k]=testFakeImaged;
        }

        String savePath="output_cgan".concat(File.separator).concat(String.valueOf(iterationCounter));

        VisualisationUtils.saveAsImage(testSamples,savePath);

        VisualisationUtils.mnistVisualize(testSamples);
    }

    /**
     * 训练判别器
     * @param generator
     * @param discriminator
     * @param realFeature
     * @param label
     * @param batchSize
     */
    private static void trainDiscriminator(ComputationGraph generator, ComputationGraph discriminator, INDArray realFeature, INDArray label, int batchSize) {
        //创建batchSize行，100列的随机数浅层空间
        INDArray latentDim = Nd4j.rand(new int[]{batchSize,  100});

        label= INDArrayUtils.toEmbeddingFormat(label);
        //用生成器生成假图片，这里的输入标签是使用随机的小批量中获取的，当然也可以自己随机生成
        INDArray fakeImaged=generator.output(latentDim,label)[0];
        //把生真实的图片和假的图按小批量的维度连接起来
        INDArray fakeAndRealImageFeature=Nd4j.concat(0,realFeature,fakeImaged);

        //把生真实的标签和假的标签按小批量的维度连接起来
        INDArray fakeAndRealLabelFeature=Nd4j.concat(0,label,label);
        //判别器输入特征
        INDArray[] discriminatorFeatures=new INDArray[] {fakeAndRealImageFeature,fakeAndRealLabelFeature};
        //判别器标签 将真假标签按N维度连接后放到标签数组中,注意标签0表示假，1表示真
        INDArray[] discriminatorLabels=new INDArray[] {Nd4j.concat(0,Nd4j.ones(batchSize, 1),Nd4j.zeros(batchSize, 1))};
        //构建多数据集（多个特征，多个标签）
        MultiDataSet discriminatorInputMultiDataSet=new MultiDataSet(discriminatorFeatures,discriminatorLabels);
        //训练判别器
        discriminator.fit(discriminatorInputMultiDataSet);
    }

    /**
     * 对抗训练
     * @param gan
     * @param batchSize
     */
    private static void trainGan(ComputationGraph gan,  int batchSize) {

        INDArray noiseLatentDim = Nd4j.rand(new int[]{batchSize, 100});

        INDArray label= RandomUtils.getRandomEmbeddingLabel(batchSize,0,9,random);

        //噪音特征
        INDArray[] noiseLatentFeature = new INDArray[]{noiseLatentDim, label};
        //这里故意把噪音的标签设为真，
        INDArray[] noiseLatentLabel = new INDArray[]{Nd4j.ones(batchSize, 1)};

        MultiDataSet ganInputMultiDataSet = new MultiDataSet(noiseLatentFeature, noiseLatentLabel);

        gan.fit(ganInputMultiDataSet);

    }

}
