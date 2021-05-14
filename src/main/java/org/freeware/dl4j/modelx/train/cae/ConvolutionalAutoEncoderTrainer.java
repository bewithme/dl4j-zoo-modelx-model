package org.freeware.dl4j.modelx.train.cae;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.freeware.dl4j.modelx.model.cae.ConvolutionalAutoEncoder;
import org.freeware.dl4j.modelx.model.gan.SGan;
import org.freeware.dl4j.modelx.train.gan.AbsGanTrainer;
import org.freeware.dl4j.modelx.utils.DataSetUtils;
import org.freeware.dl4j.modelx.utils.Sample;
import org.freeware.dl4j.modelx.utils.SplitDataSet;
import org.freeware.dl4j.modelx.utils.VisualisationUtils;
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
 *
 */
@Slf4j
public class ConvolutionalAutoEncoderTrainer extends AbsGanTrainer {


    private  static  DataNormalization dataNormalization = new ImagePreProcessingScaler(-1,1);

    private static String outputDir="output_CAE";

    public static void main(String[] args) {

        int batchSize=32;

        int imageHeight =256;

        int imageWidth =256;

        int imageChannel =3;


        String dataPath="/Users/wenfengxu/Downloads/data/mnist_png/training";

        int numClasses=10;

        ConvolutionalAutoEncoder cae= ConvolutionalAutoEncoder.builder()
                .imageChannel(imageChannel)
                .imageHeight(imageHeight)
                .imageWidth(imageWidth)
                .updater(Adam.builder()
                        .learningRate(0.0003)
                        .beta1(0.5).build()
                 )
                .build();

        //初始化无监叔判别器
        ComputationGraph caeGraph=cae.init();

        ComputationGraph encoderGraph=cae.initEncoder();

        //设置训练监听器
        setListeners(caeGraph,encoderGraph);

        log.info(caeGraph.summary());

        Nd4j.getMemoryManager().setAutoGcWindow(15 * 1000);

        DataSetIterator trainData = null;

        try {

            trainData= DataSetUtils.getDataSetIterator(dataPath,batchSize,numClasses,imageHeight,imageWidth,imageChannel);

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

                DataSet caeDataSet=new DataSet(realFeature,realFeature);

                caeGraph.fit(caeDataSet);

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
