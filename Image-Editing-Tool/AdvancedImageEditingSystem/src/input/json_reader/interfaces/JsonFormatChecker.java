package input.json_reader.interfaces;

import input.exceptions.FormatException;
import org.json.JSONObject;

public interface JsonFormatChecker {
    public void checkFormat(JSONObject jsonObject) throws FormatException;
}
