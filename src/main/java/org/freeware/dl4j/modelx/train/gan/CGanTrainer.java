package org.freeware.dl4j.modelx.train.gan;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.freeware.dl4j.modelx.model.gan.CGan;
import org.freeware.dl4j.modelx.utils.INDArrayUtils;
import org.freeware.dl4j.modelx.utils.RandomUtils;
import org.freeware.dl4j.modelx.utils.Sample;
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

    private static String outputDir="output_CGAN";

    public static void main(String[] args) {

        CGan cgan= CGan.builder().build();

        ComputationGraph generator=cgan.initGenerator();

        ComputationGraph discriminator=cgan.initDiscriminator();

        ComputationGraph gan=cgan.init();

        generator.setListeners(new PerformanceListener(10, true));

        discriminator.setListeners(new PerformanceListener(10, true));

        cgan.copyParamsFromGanToGeneratorAndDiscriminator(generator,discriminator,gan);

        log.info(gan.summary());

        Nd4j.getMemoryManager().setAutoGcWindow(15 * 1000);

        MnistDataSetIterator trainData = null;
        try {
            trainData = new MnistDataSetIterator(128, true, 42);
        } catch (IOException e) {
            e.printStackTrace();
        }

        int iterationCounter = 0;

        while (true) {

            trainData.reset();

            while (trainData.hasNext()) {

                iterationCounter++;

                DataSet dataSet=trainData.next();

                INDArray realFeature = dataSet.getFeatures();

                INDArray realLabel = dataSet.getLabels();

                int realBatchSize = (int) realFeature.size(0);

                for(int k=0;k<5;k++){
                    trainDiscriminator(generator, discriminator, realFeature, realLabel, realBatchSize);
                }

                cgan.copyParamsFromDiscriminatorToGanDiscriminator(discriminator, gan);

                trainGan( gan, realBatchSize);

                cgan.copyParamsFromGanToGenerator(generator,gan);

                visualize(generator, iterationCounter);

            }

        }



    }

    /**
     * 可视化
     * @param generator
     * @param iterationCounter
     */
    private static void visualize(ComputationGraph generator, int iterationCounter) {


        Sample[] samples=null;

        if (iterationCounter % 10== 0) {

            samples=getSamples(generator);

            VisualisationUtils.mnistVisualize(samples,"CGan");
        }
        if (iterationCounter % 1000== 0) {

            String savePath=outputDir.concat(File.separator).concat(String.valueOf(iterationCounter));

            VisualisationUtils.saveAsImage(samples,savePath);
        }
    }


    /**
     * 采样生成器
     * 9个输出
     * @param generator
     * @return
     */
    private static Sample[] getSamples(ComputationGraph generator) {

        int batchSize=1;

        Sample[] samples = new Sample[9];

        for(int k=0;k<9;k++){
            //创建batchSize行，100列的随机数浅层空间
            INDArray latentDim = Nd4j.rand(new int[]{batchSize,  100});
            //随机标签
            INDArray fakeEmbeddingLabel= RandomUtils.getRandomEmbeddingLabel(batchSize,0,9,random);
            //输出图片
            INDArray fakeImage=generator.output(latentDim,fakeEmbeddingLabel)[0];

            samples[k]=new Sample(fakeImage,String.valueOf(fakeEmbeddingLabel.toIntVector()[0]));
        }
        return samples;
    }

    /**
     * 训练判别器
     * @param generator
     * @param discriminator
     * @param realFeature
     * @param realLabel
     * @param batchSize
     */
    private static void trainDiscriminator(ComputationGraph generator, ComputationGraph discriminator, INDArray realFeature, INDArray realLabel, int batchSize) {
        //创建batchSize行，100列的随机数浅层空间
        INDArray latentDim = Nd4j.rand(new int[]{batchSize,  100});

        INDArray fakeLabel=RandomUtils.getRandomEmbeddingLabel(batchSize,0,9,random);
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
        //训练判别器
        discriminator.fit(realMultiDataSet);

        discriminator.fit(fakeMultiDataSet);
    }

    /**
     * 对抗训练
     * @param gan
     * @param batchSize
     */
    private static void trainGan(ComputationGraph gan,  int batchSize) {

        INDArray noiseLatentDim = Nd4j.rand(new int[]{batchSize, 100});

        INDArray noiseFakeLabels= RandomUtils.getRandomEmbeddingLabel(batchSize,0,9,random);
        //噪音特征
        INDArray[] noiseLatentFeature = new INDArray[]{noiseLatentDim, noiseFakeLabels};
        //这里故意把噪音的标签设为真，
        INDArray[] noiseDisLabels = new INDArray[]{Nd4j.ones(batchSize, 1)};

        MultiDataSet ganInputMultiDataSet = new MultiDataSet(noiseLatentFeature, noiseDisLabels);

        gan.fit(ganInputMultiDataSet);

    }

}
