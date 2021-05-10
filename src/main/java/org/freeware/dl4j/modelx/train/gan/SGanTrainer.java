package org.freeware.dl4j.modelx.train.gan;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.freeware.dl4j.modelx.model.gan.SGan;
import org.freeware.dl4j.modelx.utils.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;

import java.io.File;
import java.io.IOException;
import java.util.Random;


/**
 * @author wenfengxu
 * 半监督深度卷积生成对抗网络训练，该训练展示仅用1%的训练数据
 * 达到90%以上的F1Score，可拓展为其它模型的小样本数据训练。
 * 1、训练有监督判别器。
 * 2、把有监督鉴别器的参数复制给无监督判别器。
 * 3、用真实数据和生成器生成的假数据训练判别器。
 * 4、把无监督鉴别器的参数复制给有监督判别器。
 * 5、把判别器参数复制给对抗网络中的判别器，并冻结对抗网络中的判别器的参数使它不能学习。
 * 6、训练对抗网络，更新对抗网络中的生成器参数，然后把对抗网络中的生成器参数复制给生成器。
 */
@Slf4j
public class SGanTrainer extends AbsGanTrainer{


    private  static Random random=new Random(12345);

    private  static  DataNormalization dataNormalization = new ImagePreProcessingScaler(-1,1);

    private static String outputDir="output_SGAN";

    public static void main(String[] args) {

        int batchSize=32;

        int numClasses =10;

        int imageHeight =28;

        int imageWidth =28;

        int imageChannel =1;

        int latentDimLen=100;


        String dataPath="/Users/wenfengxu/Downloads/data/mnist_png/training";

        SGan sgan= SGan.builder()
                .numClasses(numClasses)
                .imageChannel(imageChannel)
                .imageHeight(imageHeight)
                .imageWidth(imageWidth)
                .updater(Adam.builder()
                        .learningRate(0.0003)
                        .beta1(0.5).build()
                 )
                .build();
        //初始化生成器网络
        ComputationGraph generator=sgan.initGenerator();
        //初始化无监督判别器
        ComputationGraph unSuperviseDiscriminator=sgan.initUnSuperviseDiscriminator();
        //初始化有监督判别器
        ComputationGraph superviseDiscriminator=sgan.initSuperviseDiscriminator();
        //初始化无监叔判别器
        ComputationGraph gan=sgan.init();
        //设置训练监听器
        setListeners(unSuperviseDiscriminator,gan,superviseDiscriminator);

        log.info(gan.summary());

        Nd4j.getMemoryManager().setAutoGcWindow(15 * 1000);
        //把数据集按比例0.001:0.99分割为训练集与测试集
        SplitDataSet splitDataSet=new SplitDataSet(0.01,dataPath,batchSize,imageHeight,imageHeight,imageChannel);
        //训练集
        DataSetIterator trainDataSetIterator = null;
        //测试集
        DataSetIterator testDataSetIterator = null;

        try {

            trainDataSetIterator=splitDataSet.getTrainDataSetIterator();

            testDataSetIterator=splitDataSet.getTestDataSetIterator();

        } catch (IOException e) {
           log.error("",e);
        }

        int iterationCounter = 0;

        while (true) {

            trainDataSetIterator.reset();

            while (trainDataSetIterator.hasNext()) {

                iterationCounter++;

                DataSet dataSet=trainDataSetIterator.next();

                INDArray realFeature = dataSet.getFeatures();

                INDArray realLabel = dataSet.getLabels();

                int realBatchSize=(int)realLabel.size(0);

                INDArray latentDim = Nd4j.rand(new int[]{realBatchSize, latentDimLen});
                //训练有监督判别器。
                superviseDiscriminator.fit(dataSet);
                //把有监督判别器的参数复制给无监督判别器
                sgan.copyParamsWithoutOutputLayer(superviseDiscriminator,unSuperviseDiscriminator);
                //训练无监督判别器
                trainDiscriminator(generator,unSuperviseDiscriminator,latentDim,realFeature,realBatchSize);
                //把无监督鉴别器的参数复制给有监督判别器。
                sgan.copyParamsWithoutOutputLayer(unSuperviseDiscriminator,superviseDiscriminator);
                //把判别器参数复制给对抗网络中的判别器，并冻结对抗网络中的判别器的参数使它不能学习
                sgan.copyParamsFromDiscriminatorToGanDiscriminator(unSuperviseDiscriminator, gan);
                //训练对抗网络
                trainGan(gan,realBatchSize,latentDim);
                //把对抗网络中的生成器参数复制给生成器
                sgan.copyParamsFromGanToGenerator(generator,gan);
                //可视化
                visualize(generator, iterationCounter);

                evaluate(superviseDiscriminator, testDataSetIterator, iterationCounter);

            }

        }

    }

    private static void evaluate(ComputationGraph superviseDiscriminator, DataSetIterator testDataSetIterator, int iterationCounter) {

        if (iterationCounter % 100== 0) {

           log.info(superviseDiscriminator.evaluate(testDataSetIterator).stats());

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

            VisualisationUtils.mnistVisualizeForConvolution2D(samples,"SGan");
        }
        if (iterationCounter % 1000== 0) {

            String savePath=outputDir.concat(File.separator).concat(String.valueOf(iterationCounter));

            VisualisationUtils.saveAsImageForConvolution2D(samples,savePath);
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
            //输出图片
            INDArray fakeImage=generator.output(latentDim)[0];
            //把图片数据恢复到0-255
            dataNormalization.revertFeatures(fakeImage);

            Sample sample=new Sample(fakeImage,"");

            samples[k]=sample;
        }
        return samples;
    }







}
