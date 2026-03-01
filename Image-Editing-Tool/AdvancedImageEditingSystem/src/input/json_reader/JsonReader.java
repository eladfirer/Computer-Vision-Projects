package input.json_reader;

import image_editor.image.DataContainer;
import input.InputReader;
import input.json_reader.interfaces.JsonDataExtractor;
import input.json_reader.interfaces.JsonFormatChecker;
import org.json.JSONObject;
import org.json.JSONTokener;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;

public class JsonReader implements InputReader {
    // interfaces for json data extractor and JsonFormatChecker.
    // we can add in a future videoJsonDataExtractor and more, and use Json reader.
    private final JsonFormatChecker formatChecker;
    private final String filePath;
    private final JsonDataExtractor dataExtractor;

    public JsonReader(
            String filePath,
            JsonFormatChecker formatChecker,
            JsonDataExtractor dataExtractor
    ) {
        this.filePath      = filePath;
        this.formatChecker = formatChecker;
        this.dataExtractor = dataExtractor;
    }

    @Override
    public DataContainer read() throws IOException {
        JSONObject json;
        try (FileInputStream fis = new FileInputStream(filePath)) {
            json = new JSONObject(new JSONTokener(fis));
        } catch (FileNotFoundException e) {
            throw new IOException("Config file not found: " + filePath, e);
        } catch (IOException e) {
            throw new IOException(e.getMessage(), e);
        }
        formatChecker.checkFormat(json);
        return dataExtractor.extractData(json);
    }
}

