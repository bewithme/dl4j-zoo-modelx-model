package org.freeware.dl4j.modelx.utils;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;
import java.io.IOException;
import java.util.Objects;
import java.util.Random;

public class SplitDataSet {



    private int batchSize;

    private int numPossibleLabels;

    private int height;

    private int width;

    private int channels;

    private ParentPathLabelGenerator parentPathLabelGenerator = new ParentPathLabelGenerator();

    private DataNormalization dataNormalization = new ImagePreProcessingScaler(-1, 1);

    private InputSplit[] inputSplit;

    public SplitDataSet(double trainInputSplitRate, String dataPath, int batchSize,  int height, int width, int channels) {
        this.batchSize = batchSize;
        this.height = height;
        this.width = width;
        this.channels = channels;
        long seed = 42;
        Random random = new Random(seed);
        FileSplit fileSplit = new FileSplit(new File(dataPath), NativeImageLoader.ALLOWED_FORMATS, random);
        //获取分类标签数
        numPossibleLabels = Objects.requireNonNull(fileSplit.getRootDir().listFiles(File::isDirectory)).length;

        /**用一个BalancedPathFilter来抽样，来实现样本均衡，提高模型性能
         需要注意的是，BalancedPathFilter抽样出来的总数量=最少样本的标签对应的样本数*标签数
         例如，有4个文件夹a,b,c,d对应的样本数量为5,10,15,20使用BalancedPathFilter之后
         抽出来的样本总数量=5*4=20个，而不是5+10+15+20=50个
         **/
        BalancedPathFilter balancedPathFilter = new BalancedPathFilter(random, null, parentPathLabelGenerator);
        inputSplit = fileSplit.sample(balancedPathFilter, trainInputSplitRate, 1 - trainInputSplitRate);

    }

    private DataSetIterator getDataSetIterator(InputSplit inputSplit) throws IOException {
        //图片读取器
        ImageRecordReader imageRecordReader = new ImageRecordReader(height, width, channels, parentPathLabelGenerator);
        //初始化读取器
        imageRecordReader.initialize(inputSplit, null);
        //创建训练数据集迭代器
        DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(imageRecordReader, batchSize, 1, numPossibleLabels);
        //数据归一化
        dataNormalization.fit(dataSetIterator);
        //数据归一化
        dataSetIterator.setPreProcessor(dataNormalization);

        return dataSetIterator;

    }

    public DataSetIterator getTrainDataSetIterator() throws IOException{

        return getDataSetIterator( inputSplit[0]);
    }

    public DataSetIterator getTestDataSetIterator() throws IOException{

        return getDataSetIterator( inputSplit[1]);
    }

}
