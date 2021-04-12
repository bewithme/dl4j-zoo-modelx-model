import lombok.extern.slf4j.Slf4j;
import org.freeware.dl4j.modelx.utils.INDArrayUtils;
import org.freeware.dl4j.modelx.utils.YoloUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


@Slf4j
public class YoloUtilsTest {






    public  static  void main(String[] args){


        testGradientOfOfGIou();
    }


    private static void testGradientOfOfGIou() {

        INDArray labels= Nd4j.create(1,1,1,1,4);


        labels.put(INDArrayUtils.getLastDimensionPointZero(labels.shape()),21);

        labels.put(INDArrayUtils.getLastDimensionPointOne(labels.shape()),45);

        labels.put(INDArrayUtils.getLastDimensionPointTwo(labels.shape()),103);

        labels.put(INDArrayUtils.getLastDimensionPointThree(labels.shape()),172);

        INDArray predict=  Nd4j.create(1,1,1,1,4);

        predict.put(INDArrayUtils.getLastDimensionPointZero(labels.shape()),59);

        predict.put(INDArrayUtils.getLastDimensionPointOne(labels.shape()),106);

        predict.put(INDArrayUtils.getLastDimensionPointTwo(labels.shape()),154);

        predict.put(INDArrayUtils.getLastDimensionPointThree(labels.shape()),230);



      

        INDArray gIou= YoloUtils.getGIou(predict,labels);


        log.info(gIou.toString());

    }

    private static void testGradientOfOfFocal() {

        INDArray labels= Nd4j.linspace(0,11,12);

        INDArray predict= Nd4j.linspace(3,14,12);

        INDArray loss= YoloUtils.focal(labels,predict);

        INDArray gradientOfLoss=YoloUtils.gradientOfOfFocal(labels,predict);

        log.info(loss.toString());

        log.info(gradientOfLoss.toString());
    }
}
