package input.json_reader.image;

import image_editor.image.ImageDataChecker;
import input.exceptions.FormatException;

import input.json_reader.interfaces.JsonFormatChecker;
import org.json.JSONArray;
import org.json.JSONObject;

import java.util.*;

import utils.Constants;


// Used Ai assistance in order to check JSON data.
public class ImageJsonFormatChecker implements JsonFormatChecker {

    public void checkFormat(JSONObject jsonObject) throws FormatException {
        checkInput(jsonObject);
        checkOutputAndDisplay(jsonObject);
        checkOperations(jsonObject);
    }


    private void checkOperations(JSONObject json) throws FormatException {
        JSONArray arr = json.optJSONArray(Constants.OPERATIONS);

        if (arr == null) {
            ImageDataChecker.getInstance().checkOperations(null);
        }

        // 3) Build the list of param‐maps
        List<Map<String,String>> ops = new ArrayList<>(arr.length());
        for (int i = 0; i < arr.length(); i++) {
            JSONObject opJson = arr.optJSONObject(i);
            if (opJson == null) {
                Map<String,String> map = new HashMap<>();
                map.put("","");
                ops.add(map);
                continue;
            }
            Map<String,String> map = new HashMap<>();
            for (String key : opJson.keySet()) {
                map.put(key, opJson.get(key).toString());
            }
            ops.add(map);
        }

        // 4) Delegate all the validation
        ImageDataChecker.getInstance().checkOperations(ops);
    }

    private void checkInput(JSONObject jsonObject) throws FormatException {
        String path = jsonObject.optString("input", null);
        ImageDataChecker.getInstance().checkInput(path);
    }

    private void checkOutputAndDisplay(JSONObject jsonObject) throws FormatException {
        String outPath  = jsonObject.optString("output", "").trim();
        boolean display = jsonObject.optBoolean("display", false);

        ImageDataChecker.getInstance()
                .checkOutputAndDisplay(outPath, display);
    }
}


