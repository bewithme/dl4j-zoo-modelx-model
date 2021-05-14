package org.freeware.dl4j.modelx.train.cae;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.freeware.dl4j.modelx.model.cae.ConvolutionalAutoEncoder;
import org.freeware.dl4j.modelx.model.gan.SGan;
import org.freeware.dl4j.modelx.train.AbsTrainer;
import org.freeware.dl4j.modelx.train.gan.AbsGanTrainer;
import org.freeware.dl4j.modelx.utils.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Adam;

import java.io.File;
import java.io.IOException;
import java.util.Random;


/**
 * @author wenfengxu
 *
 */
@Slf4j
public class ConvolutionalAutoEncoderTrainer extends AbsTrainer {


    private  static  DataNormalization dataNormalization = new ImagePreProcessingScaler(-1,1);

    private static String outputDir="output_CAE";

    private static  String modelSavePath="models/cae/";

    public static void main(String[] args) {

        int batchSize=64;

        int imageHeight =256;

        int imageWidth =256;

        int imageChannel =3;


        String dataPath="/Users/wenfengxu/Downloads/data/mnist_png/training";

        int numPossibleLabels= DataSetUtils.getFileDirectoriesCount(dataPath);

        ExtendedFileUtils.makeDirs(modelSavePath);

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

            trainData= DataSetUtils.getDataSetIterator(dataPath,batchSize,numPossibleLabels,imageHeight,imageWidth,imageChannel);

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

                visualize(caeGraph,realFeature,iterationCounter);

                if (iterationCounter % 10== 0) {

                    cae.copyParamsByName(caeGraph,encoderGraph);

                    saveModel(caeGraph,encoderGraph);

                }

            }

        }

    }


    /**
     * 可视化
     * @param cae
     * @param iterationCounter
     */
    private static void visualize(ComputationGraph cae,INDArray features, int iterationCounter) {

        Sample[] samples=null;

        if (iterationCounter % 10== 0) {

            samples=getSamples(cae,features);

            VisualisationUtils.visualizeForConvolution2D(samples,"CAE");

        }
        if (iterationCounter % 1000== 0) {

            String savePath=outputDir.concat(File.separator).concat(String.valueOf(iterationCounter));

            VisualisationUtils.saveAsImageForConvolution2D(samples,savePath);
        }
    }

    private static void saveModel(ComputationGraph cae,ComputationGraph encoder){

        if (!modelSavePath.endsWith(File.separator)){
            modelSavePath=modelSavePath.concat(File.separator);
        }

        String caeFileName=modelSavePath.concat("cae.zip");

        String encoderFileName=modelSavePath.concat("caeEncoder.zip");

        ExtendedFileUtils.deleteQuietly(new File(caeFileName));

        ExtendedFileUtils.deleteQuietly(new File(encoderFileName));

        try {
            ModelSerializer.writeModel(cae,new File(caeFileName),Boolean.TRUE);

            ModelSerializer.writeModel(cae,new File(encoderFileName),Boolean.TRUE);

        } catch (IOException e) {
           log.info("",e);
        }
    }


    /**
     * 采样生成器
     * 9个输出
     * @param cae
     * @return
     */
    private static Sample[] getSamples(ComputationGraph cae,INDArray features) {

        int batchSize=getBatchSize(features);

        batchSize=clipBatchSize(batchSize,3);
        //输入+输出
        int sampleLen=batchSize*2;

        Sample[] samples = new Sample[sampleLen];

        for(int k=0;k<batchSize;k++){

            INDArray inputFeatures =features.get(new INDArrayIndex[]{NDArrayIndex.point(k),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all()});

            inputFeatures=Nd4j.expandDims(inputFeatures,0);
            //输出图片
            INDArray outputImage=cae.output(inputFeatures)[0];
            //把图片数据恢复到0-255
            dataNormalization.revertFeatures(inputFeatures);

            dataNormalization.revertFeatures(outputImage);

            Sample sampleInput=new Sample(inputFeatures,"input");

            Sample sampleOutput=new Sample(outputImage,"output");

            samples[k*2]=sampleInput;

            samples[k*2+1]=sampleOutput;
        }
        return samples;
    }






}
