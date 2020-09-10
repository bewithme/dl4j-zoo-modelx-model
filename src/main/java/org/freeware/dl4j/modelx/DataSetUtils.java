package org.freeware.dl4j.modelx;

import lombok.extern.slf4j.Slf4j;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.Random;

@Slf4j
public class DataSetUtils {





      public static DataSetIterator getDataSetIterator(String dataPath, int batchSize, int numPossibleLabels,int height,int width,int channels) throws IOException {

        Random random = new Random(123456);

        ParentPathLabelGenerator parentPathLabelGenerator = new ParentPathLabelGenerator();

        ImageRecordReader imageRecordReader = new ImageRecordReader(height, width, channels, parentPathLabelGenerator);

        FileSplit trainSplit = new FileSplit(new File(dataPath), NativeImageLoader.ALLOWED_FORMATS, random);

        imageRecordReader.initialize(trainSplit);

        //标签索引，对于图片分类，固定为1
        int labelIndex=1;
        //创建训练集记录读取器数据迭代器
        DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(imageRecordReader, batchSize,labelIndex,numPossibleLabels);

        DataNormalization imageScaler = new ImagePreProcessingScaler();

        imageScaler.fit(dataSetIterator);

        dataSetIterator.setPreProcessor(imageScaler);

        return dataSetIterator;
    }

    /**
     * 获取指定路径下文件夹数量
     * @param dataPath
     * @return
     */
    public  static int getFileDirectoriesCount(String dataPath) {

        File file = new File(dataPath);
        String[] directories = file.list(new FilenameFilter() {
            @Override
            public boolean accept(File current, String name) {
                File currentFile=new File(current, name);
                return currentFile.isDirectory()&&!currentFile.getName().startsWith(".");
            }
        });
        log.info("count file directories from:"+dataPath);
        log.info("total labels is:"+directories.length);
        return directories.length;
    }
}
