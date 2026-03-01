package input.json_reader.image;

import image_editor.image.DataContainer;
import image_editor.image.Image;
import input.json_reader.interfaces.JsonDataExtractor;
import operations.Operation;
import operations.OperationFactory;
import input.exceptions.DataException;
import org.json.JSONArray;
import org.json.JSONObject;
import output.OutputWriter;
import output.OutputWriterFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

// Used Ai assistance in order to extract JSON data.
public class ImageJsonDataExtractor implements JsonDataExtractor {
    public DataContainer extractData(JSONObject jsonObject) throws DataException {
        try {
            // 1. Load image from input path
            String inputPath = jsonObject.getString("input");
            Image image = new Image(inputPath);

            // 2. Extract operations
            List<Operation> operations = new ArrayList<>();
            JSONArray opArray = jsonObject.getJSONArray("operations");

            for (int i = 0; i < opArray.length(); i++) {
                JSONObject opJson = opArray.getJSONObject(i);


                String type = opJson.getString("type").toLowerCase();

                Map<String,String> params = new HashMap<>();
                for (String key : opJson.keySet()) {
                    params.put(key, opJson.get(key).toString());
                }

                Operation op = OperationFactory.createOperation(type, params);
                operations.add(op);
            }

            // 3. Build real OutputWriter instances
            List<OutputWriter> outputs = new ArrayList<>();

            // display (no path required)
            if (jsonObject.optBoolean("display", false)) {
                outputs.add(OutputWriterFactory.createWriter("display", null));
            }

            // file output (if present and non-empty)
            String outPath = jsonObject.optString("output", "").trim();
            if (!outPath.isEmpty()) {
                outputs.add(OutputWriterFactory.createWriter("file", outPath));
            }
            return new DataContainer(image, operations, outputs);

        } catch (IOException e) {
            throw new DataException("Failed to load image: " + e.getMessage());
        }
    }
}
