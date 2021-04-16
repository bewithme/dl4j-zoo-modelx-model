package org.freeware.dl4j.modelx.utils;

import java.util.Random;

public class RandomUtils {


    /**
     * 随机缩放
     * @param scale
     * @return
     */
    public static float randScale(float scale, Random random){

        scale=randomUniform(1.f,scale,random);

        if(random.nextInt(2)==0){
            return scale;
        }
        return 1.0f/scale;
    }

    /**
     * 获取随机数
     * @param min
     * @param max
     * @return
     */
    public static float randomUniform(float min, float max, Random random){

        return min + ((max - min) * random.nextFloat());
    }


    /**
     * 获取随机数
     * @param min
     * @param max
     * @return
     */
    public static int randomUniform(int min,int max,Random random){
        int bound=max - min+1;
        if(bound<=0){
            return 0;
        }
        int randomInt=min+random.nextInt(bound);
        return randomInt;
    }
}
