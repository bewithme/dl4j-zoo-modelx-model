package org.freeware.dl4j.modelx.utils;


import com.fasterxml.jackson.core.JsonGenerationException;
import com.fasterxml.jackson.core.JsonParseException;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.io.StringWriter;
import java.io.Writer;


public class JsonUtils {

	public static <T> T jsonToObject(String json, Class<T> c){  
	    T o = null;  
	    try{  
	        o = new ObjectMapper().readValue(json, c);  
	    } catch (Exception e){  
	        e.printStackTrace();
	    }  
	    return o;  
	}  
	
	public static String objectToJson(Object o){  
        ObjectMapper om = new ObjectMapper();  
        Writer w = new StringWriter();  
        String json = null;  
        try {  
        om.writeValue(w, o);  
            json = w.toString();  
            w.close();  
        } catch (Exception e) {  
            e.printStackTrace();
        }  
        return json;  
    }  
	
	public static <T> T jsonToObjectThrowsException(String json, Class<T> c) throws JsonParseException, JsonMappingException, IOException{  
	    T o = new ObjectMapper().readValue(json, c);  
	    return o;  
	}  
	
	public static String objectToJsonThrowsException(Object o) throws JsonGenerationException, JsonMappingException, IOException{  
        ObjectMapper om = new ObjectMapper();  
        Writer w = new StringWriter();  
        String json = null;  
        om.writeValue(w, o);  
        json = w.toString();  
        w.close();  
        return json;  
    }  
}
