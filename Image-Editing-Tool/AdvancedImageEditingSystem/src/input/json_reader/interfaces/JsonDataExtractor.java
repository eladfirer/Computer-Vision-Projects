package input.json_reader.interfaces;

import image_editor.image.DataContainer;
import input.exceptions.DataException;
import org.json.JSONObject;


public interface JsonDataExtractor {
    public DataContainer extractData(JSONObject jsonObject) throws DataException;
}
